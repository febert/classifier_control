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
        accuml_fractions = torch.ones(self._hp.batch_size*2, device=self._hp.device)
        for c in self.tdist_classifiers:
            outdict = c(inputs)
            accuml_fractions = accuml_fractions*outdict.fraction.squeeze()
            c.out_sigmoid = accuml_fractions
            outdict.logits = torch.log(accuml_fractions)

            model_output.append(outdict)
        return model_output

from classifier_control.classifier.models.base_tempdistclassifier import BaseTempDistClassifierTestTime
from classifier_control.classifier.models.variants.single_tempdistclassifier_monotoncity import TesttimeSingleTempDistClassifier

class MonotonicityBaseTempDistClassifierTestTime(MonotonicityBaseTempDistClassifier, BaseTempDistClassifierTestTime):
    def __init__(self, overrideparams, logger=None):
        super(MonotonicityBaseTempDistClassifier, self).__init__(overrideparams, logger)
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

    def singletempdistclassifier(self):
        return TesttimeSingleTempDistClassifier
