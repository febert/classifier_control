from contextlib import contextmanager
import numpy as np
import pdb
import torch
import torch.nn as nn
import kornia
import torch.nn.functional as F
import cv2
import copy
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.models.q_function import QFunction
from classifier_control.classifier.models.multistep_dynamics import MultistepDynamics


class ModelDistanceCotrainer(BaseModel):
    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = copy.deepcopy(overrideparams)
        self.distance_function = QFunction(self.get_submodule_params(self.overrideparams, 'distance_params'), logger)
        self.dynamics_model = MultistepDynamics(self.get_submodule_params(self.overrideparams, 'model_params'), logger)
        self.override_defaults(self.overrideparams)  # override defaults with config file
        self.postprocess_params()

        assert self._hp.batch_size != -1  # make sure that batch size was overridden

    def init_optimizers(self, hp):
        self.distance_function.init_optimizers(hp)
        self.dynamics_model.init_optimizers(hp)
        self.optimizer = self.distance_function.optimizer

    def get_submodule_params(self, overrideparams, param_name):
        module_params = overrideparams.pop(param_name)
        module_params['batch_size'] = overrideparams['batch_size']
        module_params['device'] = overrideparams['device']
        module_params['data_conf'] = overrideparams['data_conf']
        return module_params

    def _default_hparams(self):
        default_dict = AttrDict({
            'consistency_weight_distance': 1.0,
            'consistency_weight_model': 0.05,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        model_output = self.dynamics_model(inputs)

        consistency_input = {
            'images': torch.cat([model_output[0] * 2 - 1, self.dynamics_model.consistency_s_t1, self.dynamics_model.consistency_goals], dim=1),
            'actions': self.dynamics_model.consistency_actions,
            't0': self.dynamics_model.consistency_t0,
            't1': self.dynamics_model.consistency_t1,
            'tg': self.dynamics_model.consistency_tg,
        }

        distance_output = self.distance_function(inputs, consistency_input)
        return model_output, distance_output

    def optim_step(self, output, losses):
        self.distance_function.optim_step(output[1], losses.distance_loss, keep_graph=True)
        self.dynamics_model.optim_step(output[0], losses.dynamics_loss)
        return losses

    def loss(self, model_output):
        losses = AttrDict()
        dynamics_output, distance_output = model_output
        dynamics_loss = self.dynamics_model.loss(dynamics_output)
        distance_loss = self.distance_function.loss(distance_output)
        losses.dynamics_loss = dynamics_loss
        losses.distance_loss = distance_loss
        batch_size = distance_output[0].shape[0]
        losses.dynamics_max_q_loss = 0
        for q_network_output in distance_output:
            losses.dynamics_max_q_loss -= q_network_output[-batch_size//3:].mean()
        losses.dynamics_max_q_loss *= self._hp.consistency_weight_model
        losses.total_loss = losses.dynamics_max_q_loss + losses.dynamics_loss.total_loss + losses.distance_loss.total_loss
        return losses

    #def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        #self.dynamics_model._log_outputs(model_output[0], inputs, losses, step, log_images, phase)
        #self.distance_function._log_outputs(model_output[1], inputs, losses, step, log_images, phase)

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        # Log generally useful outputs
        self._log_losses(losses, step, phase)
        if phase == 'train':
            self.log_gradients(step, phase)
        self.dynamics_model._log_outputs(model_output[0], inputs, losses, step, log_images, phase)
        self.distance_function._log_outputs(model_output[1], inputs, losses, step, log_images, phase)

    def get_device(self):
        return self._hp.device


def select_indices(tensor, indices):
    new_images = []
    for b in range(tensor.shape[0]):
        new_images.append(tensor[b, indices[b]])
    tensor = torch.stack(new_images, dim=0)
    return tensor


class ModelDistanceCotrainerTestTime(ModelDistanceCotrainer):
    def __init__(self, overrideparams, logger=None):
        super(ModelDistanceCotrainerTestTime, self).__init__(overrideparams, logger)
        if self._hp.classifier_restore_path is not None:
            checkpoint = torch.load(self._hp.classifier_restore_path, map_location=self._hp.device)
            self.load_state_dict(checkpoint['state_dict'])

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('classifier_restore_path', None)
        return parent_params

    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        pass

    def forward(self, inputs):
        out = super().forward(inputs)
        return out.detach().cpu().numpy()
