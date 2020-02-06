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
        accuml_fractions = None
        for c in self.tdist_classifiers:
            outdict = c(inputs)
            if accuml_fractions is None:
                accuml_fractions = torch.ones(outdict.fraction.shape[0], device=self._hp.device)
            accuml_fractions = accuml_fractions*outdict.fraction.squeeze()
            c.out_sigmoid = accuml_fractions
            outdict.logits = torch.log(accuml_fractions)
            outdict.out_sigmoid = accuml_fractions
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

    @property
    def singletempdistclassifier(self):
        return TesttimeSingleTempDistClassifier

    def forward(self, inputs):
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

def softmax(array, axis=0):
    exp = np.exp(array)
    exp = exp/(np.sum(exp, axis=axis)[:, None] + 1e-5)
    return exp

