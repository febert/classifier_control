from contextlib import contextmanager
import numpy as np
import pdb
import torch
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.models.single_tempdistclassifier import SingleTempDistClassifier


class BaseTempDistClassifier(BaseModel):
    def __init__(self, overrideparams, logger):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self.override_defaults(overrideparams)  # override defaults with config file
        self.postprocess_params()
        self.logger = logger

        assert self._hp.batch_size != -1   # make sure that batch size was overridden

        self.tdist_classifiers = []
        self.build_network()
        self._use_pred_length = False


    def _default_hparams(self):
        default_dict = AttrDict({
            'ngf': 4,  # number of feature maps in shallowest level
            'ndist_max': 10, # maximum temporal distance to classify
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
            self.tdist_classifiers = nn.ModuleList(self.tdist_classifiers)

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
        for i_cl, cl in enumerate(self.tdist_classifiers):
            setattr(losses, 'tdist{}'.format(cl.tdist), cl.loss(inputs, model_output[i_cl]))

        # compute total loss
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses


class BaseTempDistClassifierTestTime(BaseModel):
    pass
    #todo!!
