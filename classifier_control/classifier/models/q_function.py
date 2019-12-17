from contextlib import contextmanager
import numpy as np
import pdb
import torch
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
import torch.nn.functional as F

from classifier_control.classifier.models.base_model import BaseModel
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
            'nz_enc': 64,
            'classifier_restore_path':None,  # not really needed here.,
            'low_dim':False,
            'gamma':0.0
            
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
        tlen = inputs.demo_seq_images.shape[1]
        pos_pairs, neg_pairs, pos_act, neg_act = self.sample_image_triplet_actions(inputs.demo_seq_images, inputs.actions, tlen, 1, inputs.states[:, :,  :2])
        self.images = torch.cat([pos_pairs, neg_pairs], dim=0) 
        if self._hp.low_dim:
            image_0 = self.images[:, :2]
            image_g =  self.images[:, 4:]
        else:
            image_0 = self.images[:, :3]
            image_g =  self.images[:, 6:]
            
        image_pairs = torch.cat([image_0, image_g], dim=1)
        acts = torch.cat([pos_act, neg_act], dim=0)
        self.acts = acts
        
        qval = self.qnetwork(image_pairs, acts)
        return qval
    
    def sample_image_triplet_actions(self, images, actions, tlen, tdist, states):
        
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
        tg = [np.random.randint(t0[b] + tdist + 1, tlen, 1) for b in range(self._hp.batch_size)]
        tg = np.array(tg).squeeze()
        t0, t1, tg = torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(tg)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)
        im_tg = select_indices(images, tg)
        s_t0 = select_indices(states, t0)
        s_t1 = select_indices(states, t1)
        s_tg = select_indices(states, tg)
        neg_act = select_indices(actions, t0)
        self.neg_pair = torch.stack([im_t0, im_tg], dim=1)
        if self._hp.low_dim:
            self.neg_pair_cat = torch.cat([s_t0, s_t1, s_tg], dim=1)
        else:
            self.neg_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        # one means within range of tdist range,  zero means outside of tdist range
        self.labels = torch.cat([torch.ones(self._hp.batch_size), torch.zeros(self._hp.batch_size)])

        return self.pos_pair_cat, self.neg_pair_cat, pos_act, neg_act


    def loss(self, inputs, model_output):
        if self._hp.low_dim:
            image_pairs = self.images[:, 2:]
        else:
            image_pairs = self.images[:, 3:]
            
        qs = []
        for ns in range(100):
            actions = torch.FloatTensor(model_output.size(0), 2).uniform_(-1, 1).cuda()
            targetq = self.target_qnetwork(image_pairs, actions)
            qs.append(targetq)
        qs = torch.stack(qs)
        lb = self.labels.to(self._hp.device)
        
        losses = AttrDict()
        target = lb + self._hp.gamma * torch.max(qs, 0)[0].squeeze()
#         print("______________")
#         print(self.images[:5])
#         print(self.acts[:5])
#         print(model_output[:5])
#         print(target[:5])
        
#         print(self.images[-5:])
#         print(self.acts[-5:])
#         print(model_output[-5:])
#         print(target[-5:])
#         print("______________")
        losses.total_loss = F.mse_loss(target, model_output.squeeze()) 
        
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        return losses
    
    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        print(self.pos_pair_cat.shape, self.neg_pair_cat.shape)
        print(model_output.shape, self.pos_pair.shape, self.neg_pair.shape)
        if np.random.uniform() < 0.9:
          return
        
        qvals = []
        with torch.no_grad():
            print("################################################")
            for x in np.arange(-0.3, 0.3, 0.02):
                for y in np.arange(-0.3, 0.3, 0.02):
                    state = torch.FloatTensor([x,y]).cuda().unsqueeze(0).repeat(self.pos_pair_cat.size(0), 1)
                    state = torch.cat([state, self.pos_pair_cat[:,4:]], 1)
                    qs = []
                    for ns in range(100):
                        actions = torch.FloatTensor(self.pos_pair_cat.size(0), 2).uniform_(-1, 1).cuda()
                        targetq = self.target_qnetwork(state, actions)
                        qs.append(targetq)
#                     qs, _ = torch.stack(qs).max(0)
                    qs = torch.stack(qs).mean(0)
                    print(state[0], qs[0])
                    qvals.append(qs)
                    
            qvals = torch.stack(qvals)
        print("################################################")
        print(qvals.shape)
        qvals = qvals.view(30, 30, 32)
#         print(self.pos_pair_cat[0,4:])
#         print(qvals[:,:,0])
#         assert(False)
#         qvals -= qvals.min()
#         qvals /= qvals.max()
        
#         import matplotlib.pyplot as plt
#         import seaborn as sns
#         import io
# #         import tensorflow as tf
#         for i in range(self.pos_pair_cat.size(0)):
#             sns.heatmap(qvals[:, :, i].cpu().numpy())
#             plt.draw()

#             # Now we can save it to a numpy array.
#             data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#             data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# #             buf = io.BytesIO()
# #             plt.savefig(buf, format='png')
# #             buf.seek(0)
# #             image = tf.image.decode_png(buf.getvalue(), channels=3)
#             print(data.shape)
        
#         assert(False)
            
        
#         assert(False)

        if log_images:
            self._logger.log_single_tdist_classifier_image(self.pos_pair, self.neg_pair, model_output.squeeze(),
                                                          'tdist{}'.format("Q"), step, phase)
            self._logger.log_heatmap_image(self.pos_pair, qvals, model_output.squeeze(),
                                                          'tdist{}'.format("Q"), step, phase)

    def get_device(self):
        return self._hp.device


    
def select_indices(tensor, indices):
    new_images = []
    for b in range(tensor.shape[0]):
        new_images.append(tensor[b, indices[b]])
    tensor = torch.stack(new_images, dim=0)
    return tensor

class QFunctionTestTime(QFunction):
    def __init__(self, overrideparams, logger=None):
        super(QFunctionTestTime, self).__init__(overrideparams, logger)
        checkpoint = torch.load(self._hp.classifier_restore_path, map_location=self._hp.device)
        self.load_state_dict(checkpoint['state_dict'])

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('classifier_restore_path', None)
        return parent_params
