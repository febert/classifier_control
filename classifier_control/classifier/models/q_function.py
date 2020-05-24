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

        self.tdist_classifiers = []
        self.build_network()
        self._use_pred_length = False
        

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
        self.qnetwork = q_network_type(self._hp)
        with torch.no_grad():
            self.target_qnetwork = q_network_type(self._hp)

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        #### Train vs Test
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
                image_0 = self.images[:, :self._hp.state_size]
                image_g = self.images[:, 2 * self._hp.state_size:]
            else:
                image_0 = self.images[:, :3]
                image_g = self.images[:, 6:]

            image_pairs = torch.cat([image_0, image_g], dim=1)
            if self._hp.true_negatives:
                acts = torch.cat([pos_act, neg_act, tn_act], dim=0)
            else:
                acts = torch.cat([pos_act, neg_act], dim=0)
            self.acts = acts

            qval = self.qnetwork(image_pairs, acts)
        else:
            qs = []
            if self._hp.low_dim:
                if self._hp.state_size == 18:
                    inputs['current_state'] = self.get_arm_state(inputs['current_state'])
                    inputs['goal_state'] = self.get_arm_state(inputs['goal_state'])

                image_pairs = torch.cat([inputs["current_state"], inputs['goal_state']], dim=1)
            else:
                image_pairs = torch.cat([inputs["current_img"], inputs["goal_img"]], dim=1)

            if 'actions' in inputs:
                qs = self.qnetwork(image_pairs, inputs['actions'])
                return qs.detach().cpu().numpy()

            with torch.no_grad():
                for ns in range(100):
                    #actions = (torch.FloatTensor(image_pairs.size(0), self._hp.action_size).normal_() * np.array(self._hp.action_stds, dtype='float32')).cuda()
                    actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).uniform_(
                        *self._hp.action_range).cuda()
                    targetq = self.target_qnetwork(image_pairs, actions).detach()
                    qs.append(targetq)
            qs = torch.stack(qs)
            qval = torch.max(qs, 0)[0].squeeze()
            qval = qval.detach().cpu().numpy()
        return qval

    def get_arm_state(self, states):
        if len(states.shape) == 2:
            return torch.cat((states[:, :9], states[:, 15:24]), axis=1)
        return torch.cat((states[:, :, :9], states[:, :, 15:24]), axis=2)

    def sample_image_triplet_actions(self, images, actions, tlen, tdist, states):
        if self._hp.state_size == 18:
            states = self.get_arm_state(states)
        else:
            states = states[:, :, :self._hp.state_size]

        # get positives:
        t0 = np.random.randint(0, tlen - tdist - 1, self._hp.batch_size)
        t1 = t0 + 1
        tg = t0 + 1 + np.random.randint(0, tdist, self._hp.batch_size)
        t0, t1,  tg = torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(tg)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)
        im_tg = select_indices(images, tg)
        s_t0 = select_indices(states, t0)
        s_t1 = select_indices(states, t1)
        s_tg = select_indices(states, tg)
        pos_act = select_indices(actions, t0)
        self.pos_pair = torch.stack([im_t0, im_tg], dim=1)
        if self._hp.low_dim:
            self.pos_pair_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
        else:
            self.pos_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        # get negatives:
        t0 = np.random.randint(0, tlen - tdist - 1, self._hp.batch_size)
        t1 = t0 + 1
        num_fs_goal = int(self._hp.fs_goal_prop * self._hp.batch_size)
        bitmask = np.array([0] * num_fs_goal + [1] * (self._hp.batch_size-num_fs_goal))

        # 1 in bitmask means we will use an intermediate state as goal, 0 means use the final state of the trajectory
        tg = [np.random.randint(t0[b] + tdist + 1, tlen, 1).squeeze() if bitmask[b] else tlen-1 for b in range(self._hp.batch_size)]
        tg = np.array(tg).squeeze()
        t0, t1, tg = torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(tg)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)
        if self._hp.random_relabel:
            im_tg = select_indices(images, tg, batch_offset=1)
        else:
            im_tg = select_indices(images, tg)
        s_t0 = select_indices(states, t0)
        s_t1 = select_indices(states, t1)
        if self._hp.random_relabel:
            s_tg = select_indices(states, tg, batch_offset=1)
        else:
            s_tg = select_indices(states, tg)
        neg_act = select_indices(actions, t0)
        self.neg_pair = torch.stack([im_t0, im_tg], dim=1)
        if self._hp.low_dim:
            self.neg_pair_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
        else:
            self.neg_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        if self._hp.true_negatives:
            t0 = np.random.randint(0, tlen - tdist - 1, self._hp.batch_size)
            t1 = t0 + 1
            # 1 in bitmask means we will use an intermediate state as goal, 0 means use the final state of the trajectory
            tg = [np.random.randint(t0[b] + tdist + 1, tlen, 1) for b in range(self._hp.batch_size)]
            tg = np.array(tg).squeeze()
            t0, t1, tg = torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(tg)

            im_t0 = select_indices(images, t0)
            im_t1 = select_indices(images, t1)
            im_tg = select_indices(images, tg, batch_offset=1)
            s_t0 = select_indices(states, t0)
            s_t1 = select_indices(states, t1)
            s_tg = select_indices(states, tg, batch_offset=1)
            true_neg_act = select_indices(actions, t0)
            self.true_neg_pair = torch.stack([im_t0, im_tg], dim=1)
            if self._hp.low_dim:
                self.true_neg_pair_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
            else:
                self.true_neg_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        # one means within range of tdist range,  zero means outside of tdist range
        #neg_labels = torch.where(torch.norm(s_t1-s_tg, dim=1) < 0.05, torch.ones(self._hp.batch_size).cuda(), torch.zeros(self._hp.batch_size).cuda()).cuda()
        #self.labels = torch.cat([torch.ones(self._hp.batch_size).cuda(), neg_labels])
        if self._hp.true_negatives:
            self.labels = torch.cat([torch.ones(self._hp.batch_size), torch.zeros(2*self._hp.batch_size)])
            self.true_neg_lab = torch.cat([torch.zeros(2*self._hp.batch_size), torch.ones(self._hp.batch_size)])
        else:
            self.labels = torch.cat([torch.ones(self._hp.batch_size), torch.zeros(self._hp.batch_size)])

        if self._hp.true_negatives:
            return self.pos_pair_cat, self.neg_pair_cat, self.true_neg_pair_cat, pos_act, neg_act, true_neg_act
        else:
            return self.pos_pair_cat, self.neg_pair_cat, pos_act, neg_act

    def loss(self, model_output):
        if self._hp.low_dim:
            image_pairs = self.images[:, self._hp.state_size:]
        else:
            image_pairs = self.images[:, 3:]

        if self._hp.low_dim:
            image_pairs_rep = image_pairs[None].repeat(self._hp.est_max_samples, 1,  1)  # [num_samp, B, s_dim]
            image_pairs_rep = image_pairs_rep.view(
                *[self._hp.est_max_samples * model_output.size(0)] + list(image_pairs_rep.shape[2:]))
        else:
            image_pairs_rep = image_pairs[None].repeat(self._hp.est_max_samples, 1, 1, 1, 1)  # [num_samp, B, C, H, W]
            image_pairs_rep = image_pairs_rep.view(
                *[self._hp.est_max_samples * model_output.size(0)] + list(image_pairs_rep.shape[2:]))

        #actions = ((torch.FloatTensor(self._hp.est_max_samples * model_output.size(0), self._hp.action_size).normal_()) * np.array(self._hp.action_stds, dtype='float32')).cuda()
        actions = torch.FloatTensor(self._hp.est_max_samples * model_output.size(0), self._hp.action_size).uniform_(
            *self._hp.action_range).cuda()

        targetq = self.target_qnetwork(image_pairs_rep, actions).detach()
        qs = targetq.view(self._hp.est_max_samples, model_output.size(0), -1)

        #qs = []
        #with torch.no_grad():
        #    for ns in range(100):
        #        actions = torch.FloatTensor(model_output.size(0), self._hp.action_size).uniform_(-1, 1).cuda()
        #        targetq = self.target_qnetwork(image_pairs, actions).detach()
        #        qs.append(targetq)
        #qs = torch.stack(qs)

        lb = self.labels.to(self._hp.device)

        losses = AttrDict()
        if self._hp.terminal:
            target = (lb + self._hp.gamma * torch.max(qs, 0)[0].squeeze() * (1-lb)) #terminal value
        else:
            target = lb + self._hp.gamma * torch.max(qs, 0)[0].squeeze()

        if self._hp.true_negatives:
            tn_lb = self.true_neg_lab.to(self._hp.device)
            target *= (1-tn_lb)

        if self._hp.min_q:
            # Implement minq loss
            random_q_values = self.qnetwork(image_pairs_rep, actions).view(self._hp.est_max_samples, model_output.size(0), -1)
            random_density = np.log(0.5 ** actions.shape[-1]) # log uniform density
            random_q_values -= random_density
            min_q_loss = torch.logsumexp(random_q_values, dim=0).mean()
            min_q_loss -= model_output.mean()
            losses.min_q_loss = min_q_loss * self._hp.min_q_weight

        losses.bellman_loss = F.mse_loss(target, model_output.squeeze())
        losses.total_loss = torch.stack(list(losses.values())).sum()
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())

        return losses
    
    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
      ## Heatmap Logging (WIP)
#         qvals = []
#         with torch.no_grad():
#             print("################################################")
#             for x in np.arange(-0.3, 0.3, 0.02):
#                 for y in np.arange(-0.3, 0.3, 0.02):
#                     state = torch.FloatTensor([x,y]).cuda().unsqueeze(0).repeat(self.pos_pair_cat.size(0), 1)
#                     state = torch.cat([state, self.pos_pair_cat[:,4:]], 1)
#                     qs = []
#                     for ns in range(100):
#                         actions = torch.FloatTensor(self.pos_pair_cat.size(0), 2).uniform_(-1, 1).cuda()
#                         targetq = self.target_qnetwork(state, actions)
#                         qs.append(targetq)
# #                     qs, _ = torch.stack(qs).max(0)
#                     qs = torch.stack(qs).mean(0)
#                     print(state[0], qs[0])
#                     qvals.append(qs)
                    
#             qvals = torch.stack(qvals)
#         print("################################################")
#         print(qvals.shape)
#         qvals = qvals.view(30, 30, 32)
#         print(self.pos_pair_cat[0,4:])
#         print(qvals[:,:,0])
#         assert(False)
#         qvals -= qvals.min()
#         qvals /= qvals.max()
        
        if log_images:
            if self._hp.true_negatives:

                self._logger.log_single_tdist_classifier_image(self.pos_pair, self.neg_pair, model_output.squeeze()[:self._hp.batch_size*2],
                                                               'tdist{}'.format("Q"), step, phase)
                self._logger.log_one_ex(self.true_neg_pair, model_output.squeeze()[-self._hp.batch_size:].data.cpu().numpy(), 'tdist{}'.format("Q"), step, phase, 'true_negatives')
            else:
                self._logger.log_single_tdist_classifier_image(self.pos_pair, self.neg_pair, model_output.squeeze(),
                                                          'tdist{}'.format("Q"), step, phase)
#             self._logger.log_heatmap_image(self.pos_pair, qvals, model_output.squeeze(),
#                                                           'tdist{}'.format("Q"), step, phase)

    def get_device(self):
        return self._hp.device
    

def select_indices(tensor, indices, batch_offset=0):
    new_images = []
    for b in range(tensor.shape[0]):
        new_images.append(tensor[(b+batch_offset) % tensor.shape[0], indices[b]])
    tensor = torch.stack(new_images, dim=0)
    return tensor


class QFunctionTestTime(QFunction):
    def __init__(self, overrideparams, logger=None):
        super(QFunctionTestTime, self).__init__(overrideparams, logger)
        if self._hp.classifier_restore_path is not None:
            checkpoint = torch.load(self._hp.classifier_restore_path, map_location=self._hp.device)
            self.load_state_dict(checkpoint['state_dict'])
            self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
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
        timesteps = np.log(np.clip(qvals, 1e-5, 1)) / np.log(self._hp.gamma)
        return timesteps + 1
