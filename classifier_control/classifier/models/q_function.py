import numpy as np
import torch
import torch.nn.functional as F
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.q_network import QNetwork, AugQNetwork


class QFunction(BaseModel):
    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self.override_defaults(overrideparams)  # override defaults with config file
        self.postprocess_params()

        assert self._hp.batch_size != -1   # make sure that batch size was overridden

        self.build_network()
        self._use_pred_length = False
        self.target_network_counter = 0
        self.hm_counter = 0

    @property
    def num_network_outputs(self):
        return 1

    def _default_hparams(self):
        default_dict = AttrDict({
            'use_skips':False, #todo try resnet architecture!
            'ngf': 8,
            'action_size': 2,
            'state_size': 30,
            'nz_enc': 64,
            'classifier_restore_path':None,  # not really needed here.,
            'low_dim':False,
            'gamma':0.0,
            'terminal': True,
            'random_relabel': False,
            'film': False,
            'resnet': False,
            'resnet_type': 'resnet50',
            'tile_actions': False,
            'sarsa': False,
            'update_target_rate': 1,
            'double_q': False,
            'action_range': [-1.0, 1.0],
            'action_stds': [0.6, 0.6, 0.3, 0.3],
            'random_crops': False,
            'crop_goal_ind': False,
            'num_crops': 2,
            'est_max_samples': 100,
            'fs_goal_prop': 0.0,
            'rademacher_actions': False,
            'min_q': False,
            'min_q_weight': 1.0,
            'true_negatives': False,
            'l2_rew': False,
            'object_rew_frac': 0.0,
            'not_goal_cond': False,
            'shifted_rew': False, # Use -1 0 reward
            'object_rew': False,
            'add_image_mse_rew': False,
            'sum_reward': False,
            'zero_tn_target': True,
            'cross_traj_consistency': False,
            'close_arm_negatives': False,
            'n_step': 1,
            'goal_cql': False,
            'goal_cql_weight': 1.0,
            'goal_cql_eps': 0.1,
            'goal_cql_lagrange': False,
            'sl_loss_weight': 1.0,
            'sl_loss': False,
            'sigmoid': False,
            'fix_unit_mismatch': False,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def build_network(self, build_encoder=True):
        if self._hp.random_crops:
            q_network_type = AugQNetwork
        else:
            q_network_type = QNetwork
        self.qnetwork = q_network_type(self._hp, self.num_network_outputs)
        with torch.no_grad():
            self.target_qnetwork = q_network_type(self._hp, self.num_network_outputs)
            self.target_qnetwork.eval()

    def network_out_2_qval(self, network_outputs):
        """
        :param softmax_values: Tensor of softmax values of dimension [..., num_bins]
        :return: Tensor of dimension [...] containing scalar q values.
        """
        return network_outputs.squeeze()

    def get_max_q(self, qvals):
        return qvals.max(dim=0)

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        #### Train vs Test
        if self.target_qnetwork.training:
            self.target_qnetwork.eval()

        if "demo_seq_images" in inputs.keys():
            tlen = inputs.demo_seq_images.shape[1]

            if self._hp.true_negatives:
                pos_pairs, neg_pairs, true_neg_pairs, pos_act, neg_act, tn_act = self.sample_image_triplet_actions(inputs.demo_seq_images,
                                                                                           inputs.actions, tlen, 1,
                                                                                           inputs.states)
                self.images = torch.cat([pos_pairs, neg_pairs, true_neg_pairs], dim=0)
            else:
                pos_pairs, neg_pairs, pos_act, neg_act = self.sample_image_triplet_actions(inputs.demo_seq_images,
                                                                                           inputs.actions, tlen, 1,
                                                                                           inputs.states)
                self.images = torch.cat([pos_pairs, neg_pairs], dim=0)

            if self._hp.low_dim:
                if self._hp.not_goal_cond:
                    self.images[:, 2*self._hp.state_size:] = 0
                image_0 = self.images[:, :self._hp.state_size]
                image_g = self.images[:, 2*self._hp.state_size:]
            else:
                image_0 = self.images[:, :3]
                image_g = self.images[:, 6:]

            image_pairs = torch.cat([image_0, image_g], dim=1)
            if self._hp.true_negatives:
                acts = torch.cat([pos_act, neg_act, tn_act], dim=0)
            else:
                acts = torch.cat([pos_act, neg_act], dim=0)
            self.acts = acts

            self.network_out = self.qnetwork(image_pairs, acts)
            qval = self.network_out_2_qval(self.network_out)

        else:
            if self._hp.low_dim:
                if self._hp.state_size == 18:
                    inputs['current_state'] = self.get_arm_state(inputs['current_state'])
                    inputs['goal_state'] = self.get_arm_state(inputs['goal_state'])
                if self._hp.not_goal_cond:
                    inputs['goal_state'] = torch.zeros_like(inputs['goal_state'])
                image_pairs = torch.cat([inputs["current_state"], inputs['goal_state']], dim=1)
            else:
                image_pairs = torch.cat([inputs["current_img"], inputs["goal_img"]], dim=1)

            if 'online' in inputs:
                qs = self.network_out_2_qval(self.qnetwork(image_pairs, inputs['actions']))
                return qs

            if 'actions' in inputs:
                qs = self.target_qnetwork(image_pairs, inputs['actions'])
                return qs.detach().cpu().numpy()
            qs = self.compute_action_samples(image_pairs, self.target_qnetwork, parallel=False)
            qval = self.get_max_q(self.network_out_2_qval(qs))[0]
            qval = qval.detach().cpu().numpy()

        return qval

    def get_arm_state(self, states):
        assert states.shape[-1] > 18, 'State must be full vector to get arm subset'
        if len(states.shape) == 2:
            return torch.cat((states[:, :9], states[:, 15:24]), axis=1)
        return torch.cat((states[:, :, :9], states[:, :, 15:24]), axis=2)

    def compute_lowdim_reward(self, image_pairs):
        start_im = image_pairs[:, :self._hp.state_size]
        goal_im = image_pairs[:, self._hp.state_size:]
        object_rew = 0
        if self._hp.object_rew_frac > 0:
            assert self._hp.state_size > 17, 'State must include object state to use this reward!'
            start_obj_pos, goal_obj_pos = start_im[:, 9:15], goal_im[:, 9:15]
            if self._hp.not_goal_cond:
                goal_obj_pos = torch.FloatTensor([0.2, 0.8] * 3).to(self._hp.device)
            object_rew = -torch.norm((start_obj_pos - goal_obj_pos), dim=1)

        start_arm_pos, goal_arm_pos = self.get_arm_state(start_im), self.get_arm_state(goal_im)
        if self._hp.not_goal_cond:
            goal_arm_pos = torch.FloatTensor([1.6693e+00, -4.9677e-01, -6.9320e-01,  1.6383e+00,  9.2470e-01,
         7.7113e-01,  2.2918e+00, -9.8849e-04,  1.4676e-05, -3.8879e+00,
         5.4454e-02,  1.7412e+00, -4.1425e+00, -4.0342e+00,  1.0425e+00,
         2.2351e+00, -2.7262e-02,  3.0603e-04]).to(self._hp.device)

        arm_rew = -torch.norm((start_arm_pos-goal_arm_pos)[:, :9], dim=1)
        return self._hp.object_rew_frac * object_rew + (1-self._hp.object_rew_frac) * arm_rew

    def mse_reward(self, image_pairs):
        split_ind = image_pairs.shape[1]//2
        start_im, goal_im = image_pairs[:, :split_ind], image_pairs[:, split_ind:]
        return -torch.mean((start_im - goal_im) ** 2)

    def sample_image_triplet_actions(self, images, actions, tlen, tdist, states):

        if self._hp.state_size == 18:
            states = self.get_arm_state(states)
        else:
            states = states[:, :, :self._hp.state_size]

        ###
        # Setup batch splits
        num_splits = 3 if self._hp.true_negatives else 2
        split_size = int(self._hp.batch_size / num_splits + 0.5)
        if num_splits == 3:
            pos_ims, neg_ims, tn_ims = torch.split(images, split_size)
            pos_states, neg_states, tn_states = torch.split(states, split_size)
            pos_acts, neg_acts, tn_acts = torch.split(actions, split_size)
            self.pos_bs, self.neg_bs, self.tn_bs = pos_ims.shape[0], neg_ims.shape[0], tn_ims.shape[0]
        else:
            pos_ims, neg_ims = torch.split(images, split_size)
            pos_states, neg_states = torch.split(states, split_size)
            pos_acts, neg_acts = torch.split(actions, split_size)
            self.pos_bs, self.neg_bs = pos_ims.shape[0], neg_ims.shape[0]
        #
        ###

        # get positives:
        t0 = torch.randint(0, tlen - tdist - 1, (self.pos_bs,), device=self.get_device(), dtype=torch.long)
        t1 = t0 + self._hp.n_step
        tg = t0 + 1 + torch.randint(0, tdist, (self.pos_bs,), device=self.get_device(), dtype=torch.long)
        pos_tdist = tg-t0

        self.t0, self.t1, self.tg = t0, t1, tg

        im_t0 = select_indices(pos_ims, t0)
        im_t1 = select_indices(pos_ims, t1)
        im_tg = select_indices(pos_ims, tg)
        s_t0 = select_indices(pos_states, t0)
        s_t1 = select_indices(pos_states, t1)
        s_tg = select_indices(pos_states, tg)
        pos_act = select_indices(pos_acts, t0)
        self.pos_pair = torch.stack([im_t0, im_tg], dim=1)
        if self._hp.low_dim:
            self.pos_pair_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
        else:
            self.pos_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        # get negatives:
        t0 = np.random.randint(0, tlen - tdist - 1, self.neg_bs)
        t1 = t0 + self._hp.n_step
        num_fs_goal = int(self._hp.fs_goal_prop * self.neg_bs)
        bitmask = np.array([0] * num_fs_goal + [1] * (self.neg_bs-num_fs_goal))

        # 1 in bitmask means we will use an intermediate state as goal, 0 means use the final state of the trajectory
        tg = [np.random.randint(t0[b] + tdist + 1, tlen, 1).squeeze() if bitmask[b] else tlen-1 for b in range(self.neg_bs)]
        tg = np.array(tg).squeeze()
        t0, t1, tg = torch.from_numpy(t0).to(self._hp.device), torch.from_numpy(t1).to(self._hp.device), torch.from_numpy(tg).to(self._hp.device)
        negs_tdist = tg-t0

        self.t0, self.t1, self.tg = torch.cat([self.t0, t0]), torch.cat([self.t1, t1]), torch.cat([self.tg, tg])

        im_t0 = select_indices(neg_ims, t0)
        im_t1 = select_indices(neg_ims, t1)
        im_tg = select_indices(neg_ims, tg)
        s_t0 = select_indices(neg_states, t0)
        s_t1 = select_indices(neg_states, t1)
        s_tg = select_indices(neg_states, tg)
        neg_act = select_indices(neg_acts, t0)

        self.neg_pair = torch.stack([im_t0, im_tg], dim=1)
        if self._hp.low_dim:
            self.neg_pair_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
        else:
            self.neg_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        if self._hp.cross_traj_consistency:
            im_tg = select_indices(images, tg, batch_offset=1)
            s_tg = select_indices(states, tg, batch_offset=1)
            if self._hp.low_dim:
                self.cross_traj_pair_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
            else:
                self.cross_traj_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        self.data_tdists = torch.cat([pos_tdist, negs_tdist], axis=0)

        if self._hp.true_negatives:
            t0 = np.random.randint(0, tlen - tdist - 1, self.tn_bs)
            t1 = t0 + 1
            tg = [np.random.randint(t0[b] + tdist + 1, tlen, 1) for b in range(self.tn_bs)]
            tg = np.array(tg).squeeze()
            t0, t1, tg = torch.from_numpy(t0).to(self._hp.device), torch.from_numpy(t1).to(self._hp.device), torch.from_numpy(tg).to(self._hp.device)
            self.data_tdists = torch.cat([self.data_tdists, tg-t0]) # This one doesn't really make as much sense

            self.t0, self.t1, self.tg = torch.cat([self.t0, t0]), torch.cat([self.t1, t1]), torch.cat([self.tg, tg])

            im_t0 = select_indices(tn_ims, t0)
            im_t1 = select_indices(tn_ims, t1)
            im_tg = select_indices(tn_ims, tg, batch_offset=1)
            s_t0 = select_indices(tn_states, t0)
            s_t1 = select_indices(tn_states, t1)
            if self._hp.close_arm_negatives:
                s_tg = self.select_close_arm_goals(tn_states, s_t0)
            else:
                s_tg = select_indices(tn_states, tg, batch_offset=1)
            true_neg_act = select_indices(tn_acts, t0)
            self.true_neg_pair = torch.stack([im_t0, im_tg], dim=1)
            if self._hp.low_dim:
                self.true_neg_pair_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
            else:
                self.true_neg_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        if self._hp.true_negatives:
            return self.pos_pair_cat, self.neg_pair_cat, self.true_neg_pair_cat, pos_act, neg_act, true_neg_act
        else:
            return self.pos_pair_cat, self.neg_pair_cat, pos_act, neg_act

    def select_close_arm_goals(self, states, s_t0):
        # States is [B, T, N]
        # get arm states
        arm_states = states[:, :, :9]
        tsteps = states.shape[1]
        s_t0 = s_t0[:, :9]
        out = []
        for i, state in enumerate(s_t0):
            distances = torch.norm(arm_states - state, dim=2) #distances is [B, T]
            distances = torch.cat((distances[:i], distances[i+1:]), dim=0) #remove batch that traj is from
            distances = distances.view(-1) # flatten into [(B-1)*T]
            idx = torch.argmin(distances)
            if idx >= tsteps * i:
                idx += tsteps
            out.append(states[idx//tsteps, idx % tsteps])
        return torch.stack(out, dim=0)

    def get_q_samples(self, image_pairs, network, get_acts=False, parallel=True, detach_grad=True):
        if parallel:
            image_pairs_rep = image_pairs[None] # Add one dimension
            repeat_shape = [self._hp.est_max_samples] + [1] * len(image_pairs.shape)
            image_pairs_rep = image_pairs_rep.repeat(*repeat_shape) # [num_samp, B, s_dim]
            image_pairs_rep = image_pairs_rep.view(
                *[self._hp.est_max_samples * image_pairs.shape[0]] + list(image_pairs_rep.shape[2:]))

            return self.compute_action_samples(image_pairs_rep, network, parallel=parallel, return_actions=get_acts,
                                               detach_grad=detach_grad)
        else:
            return self.compute_action_samples(image_pairs, network, parallel=parallel, return_actions=get_acts,
                                        detach_grad=detach_grad)

    def compute_action_samples(self, image_pairs, network, parallel=True, return_actions=False, detach_grad=True):
        if not parallel:
            qs = []
            actions_l = []
            with torch.no_grad():
                for _ in range(self._hp.est_max_samples):
                    # actions = (torch.FloatTensor(image_pairs.size(0), self._hp.action_size).normal_() * np.array(self._hp.action_stds, dtype='float32')).cuda()
                    actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).uniform_(
                        *self._hp.action_range).to(self._hp.device)
                    #actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).uniform_(
                    #    -0.6, 0.6).to(self._hp.device)
                    targetq = network(image_pairs, actions)
                    if detach_grad:
                        targetq = targetq.detach()
                    qs.append(targetq)
                    if return_actions:
                        actions_l.append(actions)
            qs = torch.stack(qs)
            if actions_l:
                actions = torch.stack(actions_l)
        else:
            assert image_pairs.size(0) % self._hp.est_max_samples == 0, 'image pairs should already be duplicated before calling in parallel'
            actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).uniform_(
                *self._hp.action_range).to(self.get_device())
            targetq = network(image_pairs, actions)
            if detach_grad:
                targetq = targetq.detach()
            qs = targetq.view(self._hp.est_max_samples, image_pairs.size(0)//self._hp.est_max_samples, -1)

        if return_actions:
            actions = actions.view(self._hp.est_max_samples, image_pairs.size(0)//self._hp.est_max_samples, -1)
            return qs, actions
        else:
            return qs

    def get_td_error(self, image_pairs, model_output):

        qs = self.get_q_samples(image_pairs, self.target_qnetwork, detach_grad=True)
        max_qs = self.get_max_q(self.network_out_2_qval(qs))
        ret = torch.pow(self._hp.gamma, 1.0 * (self.tg - self.t0 - 1)) * (self.t1 >= self.tg) # Compute discounted return

        terminal_flag = (self.t1 >= self.tg).type(torch.ByteTensor).to(self._hp.device)

        if self._hp.l2_rew:
            assert self._hp.low_dim, 'l2 rew should probably only be used with lowdim states!'
            l2_rew = self.compute_lowdim_reward(image_pairs) / 5
            if self._hp.sum_reward:
                ret += l2_rew
            else:
                ret = l2_rew

        # if self._hp.add_image_mse_rew:
        #     image_mse_rew = self.mse_reward(image_pairs)  # This is a _negative_ value
        #     rew += image_mse_rew / 5.0

        discount = self._hp.gamma**(1.0*self._hp.n_step)
        max_qs = max_qs[0]
        if self._hp.fix_unit_mismatch:
            max_qs = max_qs.exp()
            model_output = model_output.exp()

        if self._hp.terminal:
            target = (ret + discount * max_qs * (1 - terminal_flag))  # terminal value
        else:
            target = ret + discount * max_qs

        if self._hp.true_negatives and self._hp.zero_tn_target:
            tn_lb = torch.cat([torch.zeros(self.pos_bs + self.neg_bs), torch.ones(self.tn_bs)]).to(self.get_device())
            target *= (1 - tn_lb)

        return F.mse_loss(target, model_output)

    def loss(self, model_output):

        def get_sg_pair(t):
            if self._hp.low_dim:
                x = self._hp.state_size
            else:
                x = 3
            return torch.cat((t[:, :x], t[:, 2 * x:]), axis=1)

        if self._hp.low_dim:
            image_pairs = self.images[:, self._hp.state_size:]
        else:
            image_pairs = self.images[:, 3:]

        losses = AttrDict()

        if self._hp.cross_traj_consistency:

            same_traj_pair = get_sg_pair(self.neg_pair_cat)
            cross_traj_pair = get_sg_pair(self.cross_traj_pair_cat)

            neg_acts = self.acts[self.pos_bs:self.pos_bs+self.neg_bs]

            #losses.cross_traj_loss = torch.clamp(self.qnetwork(cross_traj_pair, neg_acts)
            #                                     - self.qnetwork(same_traj_pair, neg_acts), min=0).mean()
            ct_pair_q = self.qnetwork(cross_traj_pair, neg_acts)
            same_q = self.qnetwork(same_traj_pair, neg_acts)
            losses.cross_traj_loss = 0.005 * (ct_pair_q - same_q).mean()

        if self._hp.min_q:
            # Implement minq loss
            random_q_values = self.network_out_2_qval(self.get_q_samples(get_sg_pair(self.images), self.qnetwork, detach_grad=False))
            random_density = np.log(0.5 ** self._hp.action_size) # log uniform density
            random_q_values -= random_density
            min_q_loss = torch.logsumexp(random_q_values, dim=0) - np.log(self._hp.est_max_samples)
            min_q_loss = min_q_loss.mean()
            min_q_loss -= model_output.mean()
            losses.min_q_loss = min_q_loss * self._hp.min_q_weight

        if self._hp.goal_cql:
            if self._hp.low_dim:
                states, goals = self.images[:, :self._hp.state_size], self.images[:, 2*self._hp.state_size:]
            else:
                states, goals = self.images[:, :3], self.images[:, 6:]
            acts_rep = self.acts.repeat(self.acts.shape[0], 1)
            state_rep_shape = [1] * len(states.shape)
            state_rep_shape[0] = states.shape[0]
            states = states.repeat(*state_rep_shape)
            goals = torch.repeat_interleave(goals, goals.shape[0], dim=0)
            outer_prod = torch.cat([states, goals], dim=1)

            q_mat = self.network_out_2_qval(self.qnetwork(outer_prod, acts_rep)) # [B^2, 1]
            q_mat = torch.reshape(q_mat, (self.images.shape[0], self.images.shape[0])) #[Goals, batch]

            diags = torch.diag(q_mat)
            lse = torch.logsumexp(q_mat, dim=0)

            if self._hp.goal_cql_lagrange and hasattr(self, 'log_alpha'):
                goal_cql_weight = torch.clamp(self.log_alpha.exp(), min=0.1, max=2000000.0)
            else:
                goal_cql_weight = self._hp.goal_cql_weight

            unweighted_gcql_loss = (-diags+lse).mean()
            if self._hp.goal_cql_lagrange:
                unweighted_gcql_loss -= self._hp.goal_cql_eps
            losses.goal_cql_loss = (goal_cql_weight * unweighted_gcql_loss).squeeze()
            self.goal_cql_lagrange_loss = -1 * losses.goal_cql_loss


        if self._hp.sl_loss:
            sl_targets = torch.pow(self._hp.gamma, 1.0*(self.data_tdists-1))
            # We won't use the supervised learning loss for the out of trajectory goals
            losses.sl_loss = self._hp.sl_loss_weight * F.mse_loss(model_output[:self._hp.batch_size-self.tn_bs], sl_targets[:self._hp.batch_size-self.tn_bs])

        losses.bellman_loss = self.get_td_error(image_pairs, model_output)
        losses.total_loss = torch.stack(list(losses.values())).sum()

        if 'goal_cql_loss' in losses and not self._hp.goal_cql_lagrange:
            losses.goal_cql_loss /= self._hp.goal_cql_weight # Divide this back out so we can compare log likelihoods
        if 'min_q_loss' in losses:
            losses.min_q_loss /= self._hp.min_q_weight # Divide this back out so we can compare log likelihoods

        """ Target network lagging update """
        self.target_network_counter = self.target_network_counter + 1
        if self.target_network_counter % self._hp.update_target_rate == 0:
            self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
            self.target_qnetwork.eval()
            self.target_network_counter = 0

        return losses

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        if hasattr(self, 'log_alpha'):
            self._logger.log_scalar(self.log_alpha.exp().item(), 'alpha', step, phase)
        if log_images:
            self.hm_counter += 1

        if log_images and self._hp.low_dim and self.hm_counter == 20:
            self.hm_counter = 0
            # Log heatmaps
            heatmaps = []
            x_range = -0.3, 0.3
            y_range = -0.3, 0.3
            for i in range(10):
                state = self.images[i, :self._hp.state_size]
                vary_ind = np.random.randint(0, 3) #index of object to move
                goal_state = state.clone()
                goal_state[9+2*vary_ind:11+2*vary_ind] = torch.FloatTensor((0.0, 0.0))
                goal_state_rep = goal_state[None].repeat(101*101, 1)
                xy_range = torch.linspace(x_range[0], x_range[1], steps=101).cuda()
                outer_prod = torch.stack((xy_range.repeat(101), torch.repeat_interleave(xy_range, repeats=101, dim=0)), dim=1)
                curr_states = torch.cat((goal_state_rep[:, :9+2*vary_ind], outer_prod, goal_state_rep[:, 11+2*vary_ind:]), dim=1)

                # for x in np.linspace(x_range[0], x_range[1], num=101):
                #     for y in np.linspace(y_range[0], y_range[1], num=101):
                #         cs = goal_state.clone()
                #         cs[9+vary_ind*2:10+vary_ind*2] = torch.FloatTensor((x, y))
                #         curr_states.append(cs)
                # curr_states = torch.stack(curr_states)
                self.target_qnetwork.eval()
                image_pairs = torch.cat((curr_states, goal_state_rep), dim=1)
                qs = self.get_q_samples(image_pairs, self.target_qnetwork, detach_grad=True, parallel=False)
                max_qs = self.get_max_q(self.network_out_2_qval(qs))[0].cpu().numpy()
                #image_pairs = torch.cat((goal_state_rep, goal_state_rep), dim=1)
                #qs = self.get_q_samples(image_pairs, self.target_qnetwork, detach_grad=True, parallel=True)
                #max_qs = self.get_max_q(self.network_out_2_qval(qs))[0].cpu().numpy()

                def linspace_to_slice(min, max, num):
                    lsp = np.linspace(min, max, num)
                    delta = lsp[1] - lsp[0]
                    return slice(min, max + delta, delta)

                x, y = np.mgrid[
                    linspace_to_slice(x_range[0], x_range[1], 101),
                    linspace_to_slice(y_range[0], y_range[1], 101)
                ]

                import matplotlib.pyplot as plt
                data = np.reshape(max_qs, (101, 101)).copy()

                fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
                plt.plot(np.linspace(y_range[0], y_range[1], num=101), data[:, data.shape[0] // 2])
                ax.set(xlabel='obj y pos', ylabel='expected distance')
                fig.canvas.draw()
                slice_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                slice_image = slice_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                slice_image = np.transpose(slice_image, (2, 0, 1)) # [C, H, W]
                plt.clf()

                data = data[:-1, :-1]
                cmap = plt.get_cmap('hot')
                fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
                im = ax.pcolormesh(x, y, data, cmap=cmap)
                fig.colorbar(im, ax=ax)
                # plt.subplots_adjust(left=0.3, right=0, bottom=0.3, top=0)
                ax.set(xlabel='obj x pos', ylabel='object y pos')
                plt.tight_layout()
                fig.canvas.draw()
                hmap = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                hmap_image = hmap.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                hmap_image = np.transpose(hmap_image, (2, 0, 1)) # [C, H, W]
                plt.clf()

                heatmaps.append(np.concatenate((hmap_image, slice_image), axis=1)) # Concat heightwise
                del curr_states, image_pairs, qs, max_qs, outer_prod
                torch.cuda.empty_cache()

            heatmaps = np.stack(heatmaps)
            self._logger.log_images(heatmaps, 'heatmaps', step, phase)

        if log_images:
            if self._hp.true_negatives:

                self._logger.log_single_tdist_classifier_image(self.pos_pair, self.neg_pair, model_output[:self.pos_bs + self.neg_bs],
                                                               'tdist{}'.format("Q"), step, phase)
                self._logger.log_one_ex(self.true_neg_pair, model_output[-self.tn_bs:].data.cpu().numpy(), 'tdist{}'.format("Q"), step, phase, 'true_negatives')
            else:
                self._logger.log_single_tdist_classifier_image(self.pos_pair, self.neg_pair, model_output,
                                                          'tdist{}'.format("Q"), step, phase)
#             self._logger.log_heatmap_image(self.pos_pair, qvals, model_output.squeeze(),
#                                                           'tdist{}'.format("Q"), step, phase)

    def get_device(self):
        return self._hp.device

    def qval_to_timestep(self, qvals):
        return np.log(np.clip(qvals, 1e-10, 2)) / np.log(self._hp.gamma) + 1


def select_indices(tensor, indices, batch_offset=0):
    batch_idx = torch.arange(len(indices)).cuda()
    if batch_offset != 0:
        batch_idx = torch.roll(batch_idx, batch_offset)
    if not isinstance(indices, torch.Tensor):
        indices = torch.LongTensor(indices).cuda()
    out = tensor[batch_idx, indices]
    return out


class QFunctionTestTime(QFunction):
    def __init__(self, overrideparams, logger=None):
        super(QFunctionTestTime, self).__init__(overrideparams, logger)
        if self._hp.classifier_restore_path is not None:
            checkpoint = torch.load(self._hp.classifier_restore_path, map_location=self._hp.device)
            self.load_state_dict(checkpoint['state_dict'])
            self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
            self.target_qnetwork.eval()
        else:
            print('#########################')
            print("Warning Q function weights not restored during init!!")
            print('#########################')

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('classifier_restore_path', None)
        return parent_params
      
    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        pass
      
    def forward(self, inputs):
        qvals = super().forward(inputs)
        # Compute the log to get the units to be in timesteps
        return -qvals
        #if self._hp.l2_rew or self._hp.shifted_rew:
        #    return -qvals

        timesteps = self.qval_to_timestep(qvals)
        return timesteps
