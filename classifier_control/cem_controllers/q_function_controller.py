from visual_mpc.policy.policy import Policy
import imp
import numpy as np
from visual_mpc.policy.cem_controllers.visualizer.construct_html import save_gifs, save_html, save_img, fill_template, img_entry_html, save_imgs, save_gifs_direct, save_imgs_direct, save_img_direct, save_img, save_html_direct
from visual_mpc.video_prediction.pred_util import get_context, rollout_predictions
from collections import OrderedDict
from classifier_control.classifier.utils.DistFuncEvaluation import DistFuncEvaluation
from classifier_control.classifier.models.base_tempdistclassifier import BaseTempDistClassifierTestTime
from classifier_control.classifier.models.dist_q_function import DistQFunctionTestTime
from classifier_control.classifier.models.q_function import QFunctionTestTime

from robonet.video_prediction.testing import VPredEvaluation
import torch

LOG_SHIFT = 1e-5

import cv2

def resample_imgs(images, img_size):
    if images.shape[-1] > img_size[-1]:
        interp_type = cv2.INTER_AREA
    else:
        interp_type = cv2.INTER_CUBIC
    if len(images.shape) == 5:
        resized_images = np.zeros([images.shape[0], 1, img_size[0], img_size[1], 3], dtype=np.uint8)
        for t in range(images.shape[0]):
            resized_images[t] = \
            cv2.resize(images[t].squeeze(), (img_size[1], img_size[0]), interpolation=interp_type)[None]
        return resized_images
    elif len(images.shape) == 3:
        return cv2.resize(images, (img_size[1], img_size[0]), interpolation=interp_type)

class QFunctionController(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """

        :param ag_params: agent parameters
        :param policyparams: policy parameters
        :param gpu_id: starting gpu id
        :param ngpu: number of gpus
        """
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.agentparams = ag_params
        self.img_sz = (64, 64)
        learned_cost_testparams = {}
        learned_cost_testparams['batch_size'] = self._hp.num_samples
        learned_cost_testparams['data_conf'] = {'img_sz': self.img_sz}  #todo currently uses 64x64!!
        learned_cost_testparams['classifier_restore_path'] = self._hp.learned_cost_model_path
        if self._hp.dist_q:
            self.learned_cost = DistFuncEvaluation(DistQFunctionTestTime, learned_cost_testparams)
        else:
            self.learned_cost = DistFuncEvaluation(QFunctionTestTime, learned_cost_testparams)
        self.device = self.learned_cost.model.get_device()

        self._img_height, self._img_width = [ag_params['image_height'], ag_params['image_width']]

        self._adim = self.agentparams['adim']
        self._sdim = self.agentparams['sdim']

        self._n_cam = 1 #self.predictor.n_cam

        self._desig_pix = None
        self._goal_pix = None
        self._images = None

        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None

    def reset(self):
        self._expert_score = None
        self._images = None
        self._expert_images = None
        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None
        return super(QFunctionController, self).reset()

    def _default_hparams(self):
        default_dict = {
            'action_sample_batches': 1,
            'num_samples': 200,
            'learned_cost_model_path': None,
            'verbose_every_iter': False,
            'dist_q': True,
        }
        parent_params = super(QFunctionController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def get_best_action(self, t=None):
        resampled_imgs = resample_imgs(self._images, self.img_sz) / 255.
        input_images = ten2pytrch(resampled_imgs, self.device)[-1]
        input_images = input_images[None].repeat(self._hp.num_samples, 1, 1, 1)
        input_states = torch.from_numpy(self._states)[None].float().to(self.device).repeat(self._hp.num_samples, 1)
        #goal_img = torch.from_numpy(self._goal_image)[None].float().to(self.device).repeat(self._hp.num_samples, 1, 1, 1)
        goal_img = uint2pytorch(resample_imgs(self._goal_image, self.img_sz), self._hp.num_samples, self.device)
        actions, scores = [], []
        for i in range(self._hp.action_sample_batches):
            #x_values, y_values = np.linspace(-1, 1, 41), np.linspace(-1, 1, 41)
            #xx, yy = np.meshgrid(x_values, y_values)
            #xypairs = np.dstack([xx, yy]).reshape(-1, 2)
            #action_batch = torch.from_numpy(xypairs).float().cuda()
            action_batch = torch.FloatTensor(input_images.size(0), 4).uniform_(-0.6, 0.6).cuda()
            inp_dict = {'current_img': input_images,
                        'current_state': input_states,
                        'goal_img': goal_img,
                        'actions': action_batch}
            actions.append(action_batch.cpu().numpy())
            scores.append(self.learned_cost.predict(inp_dict))
        #import ipdb; ipdb.set_trace()
        scores = np.concatenate(scores)
        actions = np.concatenate(actions)
        #print('scores')
        #print(scores)
        #print(scores[np.argsort(scores, axis=0)[:10]])
        #print('actions')
        #print(actions[np.argsort(scores, axis=0)[:10]])
        #print('true_act_score')
        #print(scores[-1])
        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots()
        #c = ax.pcolormesh(yy, xx, scores.reshape(41, 41)/10.0, cmap='hot')
        #fig.colorbar(c, ax=ax)
        #for ind in range(len(actions)):
        #   plt.scatter(actions[ind, 0], actions[ind, 1], c="red", alpha=max(0, scores[ind]/10.0))
        #plt.savefig(f'ims/action_dist{t}.png')
        #plt.close()
        best_act = actions[np.argmin(scores)]
        print(f'best act: {best_act}')
        print(f'highest qval: {np.min(scores)}')
        return best_act

    def act(self, t=None, i_tr=None, images=None, goal_image=None, verbose_worker=None, state=None):
        self._images = images
        self._states = state[-1][:2]
        print(f'state {t}: {self._states}')
        self._verbose_worker = verbose_worker

        ### Support for getting goal images from environment
        if goal_image.shape[0] == 1:
          self._goal_image = goal_image[0]
        else:
          self._goal_image = goal_image[-1, 0]  # pick the last time step as the goal image
        #print(f'goal {t}: {self._goal_image}')
        import matplotlib.pyplot as plt
        #plt.imsave(f'ims/goal{t}.png', self._goal_image)
        #plt.imsave(f'ims/image{t}.png', self._images[-1][0])
        #self.get_best_action(t)
        return {'actions': self.get_best_action(t)}


def ten2pytrch(img, device):
    """Converts images to the [-1...1] range of the hierarchical planner."""
    img = img[:, 0]
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)

def uint2pytorch(img, num_samples, device):
    img = np.tile(img[None], [num_samples, 1, 1, 1])
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)


