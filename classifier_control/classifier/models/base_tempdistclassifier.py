from contextlib import contextmanager
import numpy as np
import torch
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
from utils import add_n_dims
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.subnetworks import ConvEncoder

from classifier_control.classifier.models.single_tempdistclassifier import SingleTempDistClassifier


class BaseTempDistClassifier(BaseModel):
    def __init__(self, overrideparams, logger):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self.override_defaults(overrideparams)  # override defaults with config file
        self.logger = logger

        assert self._hp.batch_size != -1   # make sure that batch size was overridden

        self.tdist_classifiers = []
        self.build_network()
        self._use_pred_length = False


    def _default_hparams(self):
        default_dict = AttrDict({
            'ngf': 4,  # number of feature maps in shallowest level
            'ndist_max': 10 # maximum temporal distance to classify
        })
        

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params


    def build_network(self, build_encoder=True):

        for i in range(self._hp.ndist_max):
            tdist = i + 1
            self.tdist_classifiers.append(SingleTempDistClassifier(self.overrideparams, tdist, self._logger))

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        model_output = []
        for c in self.tdist_classifiers:
            model_output.append(c(inputs))
        return model_output


    def loss(self, inputs, model_output):
        losses = AttrDict()

        losses.per_classifier_loss = {'tdist{}'.format(c.tdist): c.loss() for t, c in self.tdist_classifiers}
        # compute total loss
        losses.total_loss = torch.stack(list(losses.per_classifier_loss.values())).sum()

        return losses

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super().log_outputs(model_output, inputs, losses, step, log_images, phase)

        self.logger.log_scalars(dict, 'tdist_losses', step, phase)

        if log_images:





class BaseTempDistClassifierTestTime(BaseModel):
    pass
