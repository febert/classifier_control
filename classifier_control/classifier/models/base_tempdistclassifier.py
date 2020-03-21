import numpy as np
import pdb
import torch
from visual_mpc.policy.cem_controllers.visualizer.construct_html import save_imgs_direct
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.models.single_tempdistclassifier import SingleTempDistClassifier
from classifier_control.classifier.models.single_tempdistclassifier import TesttimeSingleTempDistClassifier
from classifier_control.classifier.utils.vis_utils import visualize_barplot_array
import os
import yaml


from classifier_control.environments.sim.pointmass_maze.simple_maze import SimpleMaze
import cv2


class BaseTempDistClassifier(BaseModel):
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
            'ndist_max': 10, # maximum temporal distance to classify
            'use_skips':False, #todo try resnet architecture!
            'ngf': 8,
            'nz_enc': 64,
            'classifier_restore_path':None  # not really needed here.
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    @property
    def singletempdistclassifier(self):
        return SingleTempDistClassifier

    def build_network(self, build_encoder=True):
        for i in range(self._hp.ndist_max):
            tdist = i + 1
            self.tdist_classifiers.append(self.singletempdistclassifier(self._hp, tdist, self._logger))
        self.tdist_classifiers = nn.ModuleList(self.tdist_classifiers)

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
#         import pdb; pdb.set_trace()
#         assert False  # only for testign

        model_output = []
        for c in self.tdist_classifiers:
            model_output.append(c(inputs))
        return model_output

    def loss(self, model_output):
        losses = AttrDict()
        for i_cl, cl in enumerate(self.tdist_classifiers):
            setattr(losses, 'tdist{}'.format(cl.tdist), cl.loss(model_output[i_cl]))

        # compute total loss
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses

    def get_device(self):
        return self._hp.device


class BaseTempDistClassifierTestTime(BaseTempDistClassifier):
    def __init__(self, overrideparams, logger=None):
        super(BaseTempDistClassifierTestTime, self).__init__(overrideparams, logger)
        if self._hp.classifier_restore_path is not None:
            checkpoint = torch.load(self._hp.classifier_restore_path, map_location=self._hp.device)
            self.load_state_dict(checkpoint['state_dict'])
        else:
            print('#########################')
            print("Warning Classifier weights not restored during init!!")
            print('#########################')

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('classifier_restore_path', None)
        return parent_params


    def forward(self, inputs):
#         self.gen_pairs()
        outputs = super().forward(inputs)

        sigmoid = []
        for i in range(self._hp.ndist_max):
            sigmoid.append(outputs[i].out_sigmoid.data.cpu().numpy().squeeze())
        self.sigmoids = np.stack(sigmoid, axis=1)
        sigmoids_shifted = np.concatenate((np.zeros([self._hp.batch_size, 1]), self.sigmoids[:, :-1]), axis=1)
        differences = self.sigmoids - sigmoids_shifted
        self.softmax_differences = softmax(differences, axis=1)
        expected_dist = np.sum((1 + np.arange(self.softmax_differences.shape[1])[None]) * self.softmax_differences, 1)

        return expected_dist

    def gen_pairs(self):
      '''HEATMAP GENERATION'''
      env_params = env_params = {
          # resolution sufficient for 16x anti-aliasing
          'viewer_image_height': 48,
          'viewer_image_width': 64,
      #     'difficulty': 'm',
      }
      env = SimpleMaze(env_params)
      env.reset()
      goalim = env.get_goal()[0] / 255.
      goalim = goalim * 2
      goalim -= 1
      goalim = cv2.resize(goalim, (64, 64), interpolation=cv2.INTER_CUBIC)
      vals = []
      trs = range(-30, 30, 2)
      ims = []
      for x in trs:
        print(x)
        v = []
        for y in trs:
          env.sim.data.qpos[0] = x / 100.0
          env.sim.data.qpos[1] = y / 100.0
          env.sim.step()
          currim = env.render()[0] / 255.
          currim = currim * 2
          currim -= 1
          currim = cv2.resize(currim, (64, 64), interpolation=cv2.INTER_CUBIC)
          v.append(currim)
        ims.append(v)
      ims = np.array(ims)
      print(ims.shape)
      ims = ims.reshape(-1, 64, 64, 3, order='F')
          
      inputs = {
        'current_img': torch.FloatTensor(ims).cuda().permute(0, 3, 1, 2), 
        'goal_img': torch.FloatTensor(goalim).cuda().unsqueeze(0).permute(0, 3, 1, 2).repeat(900, 1, 1, 1)
      }
#       qval = super().forward(inputs) 
      outputs = super().forward(inputs)

      sigmoid = []
      for i in range(self._hp.ndist_max):
        sigmoid.append(outputs[i].out_sigmoid.data.cpu().numpy().squeeze())
      self.sigmoids = np.stack(sigmoid, axis=1)
      sigmoids_shifted = np.concatenate((np.zeros([900, 1]), self.sigmoids[:, :-1]), axis=1)
      differences = self.sigmoids - sigmoids_shifted
      self.softmax_differences = softmax(differences, axis=1)
      qval = np.sum((1 + np.arange(self.softmax_differences.shape[1])[None]) * self.softmax_differences, 1)

      qv = qval.reshape(30, 30, 1)
      qv = qv.repeat(3, -1)
      qv -= qv.min()
      qv /= qv.max()
      qv = 1 - qv
      
      qv *= 255.0
      qv = cv2.resize(qv, (64, 64), interpolation=cv2.INTER_CUBIC)
      
      goalim -= goalim.min()
      goalim /= goalim.max()

      print(qv.min(), qv.max())
      print(goalim.shape, qv.shape)
      cv2.imwrite('ims/goal.png', goalim*255.0)
      cv2.imwrite('ims/h.png', qv)
      assert(False)
          

    @property
    def singletempdistclassifier(self):
        return TesttimeSingleTempDistClassifier

    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        # save classifier preds
        sel_sigmoids = self.sigmoids[visualize_indices]
        sigmoid_images = visualize_barplot_array(sel_sigmoids)
        row_name = 'sigmoid_images'
        content_dict[row_name] = save_imgs_direct(verbose_folder,
                                                  row_name, sigmoid_images)

        sel_softmax_dists = self.softmax_differences[visualize_indices]
        sigmoid_images = visualize_barplot_array(sel_softmax_dists)
        row_name = 'softmax_of_differences'
        content_dict[row_name] = save_imgs_direct(verbose_folder,
                                                  row_name, sigmoid_images)

    def vis_dist_over_traj(self, inputs, step):
        images = inputs.demo_seq_images

        cols = []
        n_ex = 10
        
        for t in range(images.shape[1]):
            outputs = self.forward({'current_img': images[:,t], 'goal_img': images[:, -1]})

            sigmoid = []
            for i in range(len(outputs)):
                sigmoid.append(outputs[i].out_sigmoid.data.cpu().numpy().squeeze())
            sigmoids = np.stack(sigmoid, axis=1)

            sig_img_t = visualize_barplot_array(sigmoids[:n_ex])

            img_column_t = []
            images_npy = images.data.cpu().numpy().squeeze()
            for b in range(n_ex):
                img_column_t.append(ptrch2uint8(images_npy[b, t]))
                img_column_t.append(np.transpose(sig_img_t[b], [2,0,1]))
            img_column_t = np.concatenate(img_column_t, axis=1)
            cols.append(img_column_t)

        image = np.concatenate(cols, axis=2)
        # cv2.imwrite('/nfs/kun1/users/febert/data/vmpc_exp/dist_over_traj.png', np.transpose(image, [1,2,0]))
        self._logger.log_image(image, 'dist_over_traj', step, phase='test')
        self._logger.log_video(np.stack(cols, axis=0), 'dist_over_traj_gif', step, phase='test')

        print('logged dist over traj')

def softmax(array, axis=0):
    exp = np.exp(array)
    exp = exp/(np.sum(exp, axis=axis)[:, None] + 1e-5)
    return exp

def ptrch2uint8(img):
    return ((img + 1)/2*255.).astype(np.uint8)
