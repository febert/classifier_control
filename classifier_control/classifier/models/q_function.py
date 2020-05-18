import numpy as np
import torch
import torch.nn.functional as F
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.q_network import QNetwork


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
            'resnet': False,
            'random_relabel': False,
            'film': False,
            'rademacher_actions': False,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def build_network(self, build_encoder=True):
        self.qnetwork = QNetwork(self._hp)
        with torch.no_grad():
            self.target_qnetwork = QNetwork(self._hp)

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
            acts = torch.cat([pos_act, neg_act], dim=0)
            self.acts = acts

            qval = self.qnetwork(image_pairs, None)
        else:
            qs = []
            if self._hp.low_dim:
                image_pairs = torch.cat([inputs["current_state"], inputs['goal_img']], dim=1)
            else:
                image_pairs = torch.cat([inputs["current_img"], inputs["goal_img"]], dim=1)

            if 'actions' in inputs:
                qs = self.qnetwork(image_pairs, inputs['actions'])
                return qs.detach().cpu().numpy()

            with torch.no_grad():
                if self._hp.rademacher_actions:
                    for action in [-1, 1]:
                        actions = torch.FloatTensor(np.full((image_pairs.size(0), self._hp.action_size), action)).cuda()
                        # actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).normal_(mean=0, std=0.3).cuda()
                        targetq = self.target_qnetwork(image_pairs, actions)
                        qs.append(targetq)
                else:
                    for ns in range(100):
                        actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).uniform_(-1, 1).cuda()
                        targetq = self.target_qnetwork(image_pairs, actions).detach()
                        qs.append(targetq)
            qs = torch.stack(qs)
            qval = torch.max(qs, 0)[0].squeeze()
            qval = qval.detach().cpu().numpy()
        return qval
    
    def sample_image_triplet_actions(self, images, actions, tlen, tdist, states):

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
        t0 = np.random.randint(0, tlen - tdist - 4, self._hp.batch_size)
        t1 = t0 + 1
        #tg = [np.random.randint(t0[b] + tdist + 1, tlen, 1) for b in range(self._hp.batch_size)]
        tg = [np.random.randint(t0[b] + tdist + 1, tlen-2, 1) for b in range(self._hp.batch_size)]
        tg = [tg[x] if abs((tg[x]-t0[x]) % 2) == 1 else tg[x]+1 for x in range(len(tg))]
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

        # one means within range of tdist range,  zero means outside of tdist range
        #neg_labels = torch.where(torch.norm(s_t1-s_tg, dim=1) < 0.05, torch.ones(self._hp.batch_size).cuda(), torch.zeros(self._hp.batch_size).cuda()).cuda()
        #self.labels = torch.cat([torch.ones(self._hp.batch_size).cuda(), neg_labels])
        self.labels = torch.cat([torch.ones(self._hp.batch_size), torch.zeros(self._hp.batch_size)])

        return self.pos_pair_cat, self.neg_pair_cat, pos_act, neg_act

    def loss(self, model_output):
        if self._hp.low_dim:
            image_pairs = self.images[:, self._hp.state_size:]
        else:
            image_pairs = self.images[:, 3:]
            
        qs = []
        with torch.no_grad():

            if self._hp.rademacher_actions:
                targetq = self.target_qnetwork(image_pairs, None)
                qs.append(targetq)
            elif self._hp.rademacher_actions:
                for action in [-1, 1]:
                    actions = torch.FloatTensor(np.full((image_pairs.size(0), self._hp.action_size), action)).cuda()
                    # actions = torch.FloatTensor(image_pairs.size(0), self._hp.action_size).normal_(mean=0, std=0.3).cuda()
                    targetq = self.target_qnetwork(image_pairs, actions)
                    qs.append(targetq)
            else:
                for ns in range(100):
                    actions = torch.FloatTensor(model_output.size(0), self._hp.action_size).uniform_(-1, 1).cuda()
                    targetq = self.target_qnetwork(image_pairs, actions).detach()
                    qs.append(targetq)
        qs = torch.stack(qs).squeeze().detach()
        lb = self.labels.to(self._hp.device)
        losses = AttrDict()
        if self._hp.terminal:
            target = lb + self._hp.gamma * torch.max(qs, 1)[0].squeeze() * (1-lb) #terminal value
        else:
            target = lb + self._hp.gamma * torch.max(qs, 0)[0].squeeze()
        corres = torch.where(self.acts == 1, model_output[:, 0], model_output[:, 1])
        #losses.total_loss = F.mse_loss(target, model_output.squeeze())
        losses.total_loss = F.mse_loss(target, corres)
        
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
