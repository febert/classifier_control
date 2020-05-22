from contextlib import contextmanager
import numpy as np
import pdb
import torch
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
import torch.nn.functional as F

from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.q_network import DistQNetwork, AugDistQNetwork

class DistQFunction(BaseModel):

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

    def _default_hparams(self):
        default_dict = AttrDict({
            'use_skips':False, #todo try resnet architecture!
            'ngf': 8,
            'action_size': 2,
            'nz_enc': 64,
            'classifier_restore_path':None,  # not really needed here.,
            'low_dim': False,
            'gamma': 0.0,
            'est_max_samples': 100,
            'resnet': False,
            'resnet_type': 'resnet50',
            'tile_actions': False,
            'sarsa': False,
            'update_target_rate': 1,
            'double_q': False,
            'action_range': [-1.0, 1.0],
            'film': False,
            'random_crops': False,
            'crop_goal_ind': False,
            'num_crops': 2,
            'num_bins': 10,
            'rademacher_actions': False
        })


        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def build_network(self, build_encoder=True):
        if self._hp.random_crops:
            q_network_type = AugDistQNetwork
        else:
            q_network_type = DistQNetwork
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
          pos_pairs, neg_pairs, pos_act, neg_act = self.sample_image_triplet_actions(inputs.demo_seq_images, inputs.actions, tlen, 1, inputs.states[:, :,  :2])
          self.images = torch.cat([pos_pairs, neg_pairs], dim=0) 
          if self._hp.low_dim:
              image_0 = self.images[:, :2]
              image_g = self.images[:, 4:]
          else:
              image_0 = self.images[:, :3]
              image_g = self.images[:, 6:]

          image_pairs = torch.cat([image_0, image_g], dim=1)
          acts = torch.cat([pos_act, neg_act], dim=0)
          self.acts = acts

          self.out_softmax = self.qnetwork(image_pairs, acts)
          ## Compute Expectation
          qval = torch.sum((1 + torch.arange(self.out_softmax.shape[1])[None]).float().to(self._hp.device) * self.out_softmax, 1)
        else:
          qs = []

          if self._hp.low_dim:
            image_pairs = torch.cat([inputs["current_state"], inputs["goal_img"]], dim=1)
          else:
            image_pairs = torch.cat([inputs["current_img"], inputs["goal_img"]], dim=1)

          if 'actions' in inputs:   # If actions are specified at test time, compute Q(s, a) instead of max_a Q(s, a)
            qs = self.target_qnetwork(image_pairs, inputs['actions'])  # qs: [B, num_bins]
            qval = torch.sum((1 + torch.arange(qs.shape[1])[None]).float().to(self._hp.device) * qs, 1)
            return qval.detach().cpu().numpy()

          with torch.no_grad():
            if self._hp.rademacher_actions:
                for action in [-1, 1]:
                  actions = torch.FloatTensor(np.full((image_pairs.size(0), self._hp.action_size), action)).cuda()
                  # actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).normal_(mean=0, std=0.3).cuda()
                  targetq = self.target_qnetwork(image_pairs, actions)
                  qs.append(targetq)
            else:
                for ns in range(self._hp.est_max_samples):
                  actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).uniform_(*self._hp.action_range).cuda()
                  #actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).normal_(mean=0, std=0.3).cuda()
                  targetq = self.target_qnetwork(image_pairs, actions)
                  qs.append(targetq)

          qs = torch.stack(qs)

          ## Compute Expectation
          qval = torch.sum((1 + torch.arange(qs.shape[2])[None]).float().to(self._hp.device) * qs, 2)
          ## Select corresponding target Q distribution
          qval = qval.min(0)[0].squeeze()
          qval = qval.detach().cpu().numpy()
        return qval
    
    def sample_image_triplet_actions(self, images, actions, tlen, tdist, states):
        # get positives:
        t0 = np.random.randint(0, tlen - tdist - 2, self._hp.batch_size)
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
        pos_act_t1 = select_indices(actions, t1)

        self.pos_pair = torch.stack([im_t0, im_tg], dim=1)
        if self._hp.low_dim:
            self.pos_pair_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
        else:
            self.pos_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        # get negatives:
        t0 = np.random.randint(0, tlen - tdist - 4, self._hp.batch_size)
        t1 = t0 + 1
        tg = [np.random.randint(t0[b] + tdist + 1, tlen-2, 1) for b in range(self._hp.batch_size)]
        tg = [tg[x] if abs((tg[x]-t0[x]) % 2) == 1 else tg[x]+1 for x in range(len(tg))]
        tg = np.array(tg).squeeze()
        t0, t1, tg = torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(tg)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)
        im_tg = select_indices(images, tg)
        s_t0 = select_indices(states, t0)
        s_t1 = select_indices(states, t1)
        s_tg = select_indices(states, tg)
        neg_act = select_indices(actions, t0)
        neg_act_t1 = select_indices(actions, t1)
        self.neg_pair = torch.stack([im_t0, im_tg], dim=1)
        if self._hp.low_dim:
            self.neg_pair_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
        else:
            self.neg_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        # one means within range of tdist range,  zero means outside of tdist range
        self.labels = torch.cat([torch.ones(self._hp.batch_size), torch.zeros(self._hp.batch_size)])
        self.acts_t1 = torch.cat([pos_act_t1, neg_act_t1], dim=0)

        return self.pos_pair_cat, self.neg_pair_cat, pos_act, neg_act

    def loss(self, model_output):
        if self._hp.low_dim:
            image_pairs = self.images[:, self._hp.state_size:]
        else:
            image_pairs = self.images[:, 3:]
        ## Get max_a Q (s_t+1) (Is a min since lower is better)
        qs = []
        total_actions = []

        with torch.no_grad():
            if self._hp.sarsa:
                targetq = self.target_qnetwork(image_pairs, self.acts_t1)
                qs.append(targetq)
            elif self._hp.double_q:
                for ns in range(self._hp.est_max_samples):
                    actions = torch.FloatTensor(model_output.size(0), self._hp.action_size).uniform_(*self._hp.action_range).cuda()
                    targetq = self.qnetwork(image_pairs, actions).detach() # argmax using current Q value
                    qs.append(targetq)
                    total_actions.append(actions)
            else:
                #for ns in range(self._hp.est_max_samples):
                #    actions = torch.FloatTensor(model_output.size(0), self._hp.action_size).uniform_(*self._hp.action_range).cuda()
                #    targetq = self.target_qnetwork(image_pairs, actions)
                #    qs.append(targetq)
                #    total_actions.append(actions)
                if self._hp.rademacher_actions:
                    for action in [-1, 1]:
                        actions = torch.FloatTensor(np.full((image_pairs.size(0), self._hp.action_size), action)).cuda()
                        # actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).normal_(mean=0, std=0.3).cuda()
                        targetq = self.target_qnetwork(image_pairs, actions)
                        qs.append(targetq)
                        total_actions.append(actions)
                else:
                    image_pairs_rep = image_pairs[None].repeat(self._hp.est_max_samples, 1, 1, 1, 1) # [num_samp, B, C, H, W]
                    image_pairs_rep = image_pairs_rep.view(*[self._hp.est_max_samples * model_output.size(0)] + list(image_pairs_rep.shape[2:]))
                    actions = torch.FloatTensor(self._hp.est_max_samples*model_output.size(0), self._hp.action_size).uniform_(*self._hp.action_range).cuda()
                    #actions = torch.FloatTensor(self._hp.est_max_samples*model_output.size(0), self._hp.action_size).normal_(mean=0.0, std=0.3).cuda()
                    targetq = self.target_qnetwork(image_pairs_rep, actions)
                    qs = targetq.view(self._hp.est_max_samples, model_output.size(0), -1)
                    total_actions = actions.view(self._hp.est_max_samples, model_output.size(0), -1)

        if isinstance(qs, list):
            qs = torch.stack(qs)
            total_actions = torch.stack(total_actions)
        qval = torch.sum((1 + torch.arange(qs.shape[2])[None]).float().to(self._hp.device) * qs, 2)
        ## Select corresponding target Q distribution
        ids = qval.min(0)[1]
        best_actions = torch.stack([total_actions[ids[k], k] for k in range(self._hp.batch_size*2)])
        qs = self.target_qnetwork(image_pairs, best_actions).detach()
        #for k in range(self._hp.batch_size*2):
        #  newqs.append(qs[ids[k], k])
        #qs = torch.stack(newqs)
        
        ## Shift Q*(s_t+1) to get Q*(s_t)
        shifted = torch.zeros(qs.size()).to(self._hp.device)
        shifted[:, 1:] = qs[:, :-1]
        shifted[:, -1] += qs[:, -1]
        lb = self.labels.to(self._hp.device).unsqueeze(-1)
        isg = torch.zeros((self._hp.batch_size * 2, self._hp.num_bins)).to(self._hp.device)
        isg[:,0] = 1
        
        ## If next state is goal then target should be 0, else should be shifted
        losses = AttrDict()
        target = (lb * isg) + ((1-lb) * shifted)
        
        ## KL between target and output
        log_q = self.out_softmax.clamp(1e-5, 1-1e-5).log()
        log_t = target.clamp(1e-5, 1-1e-5).log()
        losses.total_loss = (target * (log_t - log_q)).sum(1).mean()
        batched_loss = (target * (log_t - log_q)).sum(1)
        # Reduce [2*B] to [B]
        losses.per_traj_loss = (batched_loss[:self._hp.batch_size] + batched_loss[self._hp.batch_size:]) / 2

        self.target_network_counter = self.target_network_counter + 1
        if self.target_network_counter % self._hp.update_target_rate == 0:
            self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
            self.target_network_counter = 0

        return losses
    
    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        if log_images:
            self._logger.log_single_tdist_classifier_image(self.pos_pair, self.neg_pair, model_output.squeeze(),
                                                          'tdist{}'.format("Q"), step, phase)

    def get_device(self):
        return self._hp.device
    

def select_indices(tensor, indices):
    new_images = []
    for b in range(tensor.shape[0]):
        new_images.append(tensor[b, indices[b]])
    tensor = torch.stack(new_images, dim=0)
    return tensor


class DistQFunctionTestTime(DistQFunction):
    def __init__(self, overrideparams, logger=None):
        super(DistQFunctionTestTime, self).__init__(overrideparams, logger)
        checkpoint = torch.load(self._hp.classifier_restore_path, map_location=self._hp.device)
        self.load_state_dict(checkpoint['state_dict'])

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('classifier_restore_path', None)
        return parent_params
      
    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        pass
      
    def forward(self, inputs):
      qvals = super().forward(inputs)
      ## Qvals represent distance now (lower is better), so directly return cost (no negative needed)
      return qvals
