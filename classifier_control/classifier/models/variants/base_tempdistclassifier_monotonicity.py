import numpy as np
import pdb
import torch
from visual_mpc.policy.cem_controllers.visualizer.construct_html import save_imgs_direct
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
from classifier_control.classifier.models.base_tempdistclassifier import BaseTempDistClassifier
from classifier_control.classifier.models.variants.single_tempdistclassifier_monotoncity import SingleTempDistClassifierMonotone


class MonotonicityBaseTempDistClassifier(BaseTempDistClassifier):

    @property
    def singletempdistclassifier(self):
        return SingleTempDistClassifierMonotone

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        model_output = []
        accuml_fractions = torch.ones(self._hp.batch_size, device=self._hp.device)
        for c in self.tdist_classifiers:
            outdict = c(inputs)
            accuml_fractions = accuml_fractions*outdict.fraction
            c.out_sigmoid = accuml_fractions
            outdict.logits = torch.log(accuml_fractions)
            model_output.append(outdict)
        return model_output
