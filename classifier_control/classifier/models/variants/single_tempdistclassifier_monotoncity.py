import numpy as np
import copy
import torch
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
import pdb
from classifier_control.classifier.utils.subnetworks import ConvEncoder
from classifier_control.classifier.utils.spatial_softmax import SpatialSoftmax

from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.layers import Linear


class SingleTempDistClassifierMonotone(BaseModel):

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """

        tlen = inputs.demo_seq_images.shape[1]
        pos_pairs, neg_pairs = self.sample_image_pair(inputs.demo_seq_images, tlen, self.tdist)
        image_pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        embeddings = self.encoder(image_pairs)
        embeddings = self.spatial_softmax(embeddings)

        fraction = torch.sigmoid(self.linear(embeddings))

        model_output = AttrDict(fraction=fraction, pos_pair=self.pos_pair, neg_pair=self.neg_pair)
        return model_output


    def loss(self, model_output):
        logits_ = model_output.logits[:, 0]
        return self.cross_ent_loss(logits_, self.labels.to(self._hp.device))


def select_indices(tensor, indices):
    new_images = []
    for b in range(tensor.shape[0]):
        new_images.append(tensor[b, indices[b]])
    tensor = torch.stack(new_images, dim=0)
    return tensor

class TesttimeSingleTempDistClassifier(SingleTempDistClassifierMonotone):
    def __init__(self, params, tdist, logger):
        super().__init__(params, tdist, logger)

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x channel x height x width
        :return: model_output
        """

        image_pairs = torch.cat([inputs['current_img'], inputs['goal_img']], dim=1)
        embeddings = self.encoder(image_pairs)
        embeddings = self.spatial_softmax(embeddings)
        logits = self.linear(embeddings)
        self.out_sigmoid = torch.sigmoid(logits)
        model_output = AttrDict(logits=logits, out_simoid=self.out_sigmoid)
        return model_output



