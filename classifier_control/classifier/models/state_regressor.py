from contextlib import contextmanager
from classifier_control.classifier.utils.spatial_softmax import SpatialSoftmax
from classifier_control.classifier.utils.layers import Linear
from classifier_control.classifier.utils.subnetworks import ConvEncoder
import numpy as np
import cv2
import pdb
import torch
import torch.nn.functional as F
from torchvision import models
from classifier_control.classifier.utils.logger import TdistRegressorLogger
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.models.single_tempdistclassifier import SingleTempDistClassifier
from classifier_control.classifier.models.single_tempdistclassifier import TesttimeSingleTempDistClassifier

from classifier_control.classifier.models.utils.utils import select_indices
from classifier_control.classifier.utils.resnet_module import get_resnet_encoder

class StateRegressor(BaseModel):
    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self.override_defaults(overrideparams)  # override defaults with config file
        self.postprocess_params()

        assert self._hp.batch_size != -1   # make sure that batch size was overridden

        self.tdist_classifiers = []
        self.build_network()

    def _default_hparams(self):
        default_dict = AttrDict({
            'use_skips':False, #todo try resnet architecture!
            'ngf': 8,
            'nz_enc': 64,
            'classifier_restore_path':None,  # not really needed here.
            'skips_stride': None,
            'resnet': False,
            'resnet_type': 'resnet50'
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params


    def build_network(self):
        if self._hp.resnet:
            if self._hp.resnet_type == 'resnet50':
                self.encoder = get_resnet_encoder(models.resnet50, self._hp.state_dim, freeze=False)
            elif self._hp.resnet_type == 'resnet18':
                self.encoder = get_resnet_encoder(models.resnet18, self._hp.state_dim, freeze=False)
            return
        self.encoder = ConvEncoder(self._hp)
        out_size = self.encoder.get_output_size()
        self.linear = Linear(in_dim=256, out_dim=self._hp.state_dim, builder=self._hp.builder)
        if self._hp.spatial_softmax:
            self.spatial_softmax = SpatialSoftmax(out_size[1], out_size[2], out_size[0])  # height, width, channel
            self.linear2 = Linear(in_dim=out_size[0]*2, out_dim=256, builder=self._hp.builder)
        else:
            self.linear2 = Linear(in_dim=out_size[0]*out_size[1]*out_size[2], out_dim=256, builder=self._hp.builder)
            self.linear3 = Linear(in_dim=256, out_dim=256, builder=self._hp.builder)
            self.linear4 = Linear(in_dim=256, out_dim=256, builder=self._hp.builder)
            self.linear5 = Linear(in_dim=256, out_dim=256, builder=self._hp.builder)

        if self._hp.use_skips:
            self.linear_skipmerge = Linear(in_dim=125696, out_dim=256, builder=self._hp.builder)

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        images = inputs.demo_seq_images
        states = inputs.states
        take_inds = np.random.randint(0, states.shape[1], states.shape[0])
        take_states = []
        take_images = []
        for i in range(states.shape[0]):
            take_states.append(inputs.states[i, take_inds[i]])
            take_images.append(images[i, take_inds[i]])
        self.labels = torch.stack(take_states)[:, :15]
        self.images = torch.stack(take_images)
        images = self.images
        model_output = self.make_prediction(images)
        return model_output

    def make_prediction(self, images):
        if self._hp.resnet:
            self.state_estimates = self.encoder(images)
            model_output = AttrDict(state_est=self.state_estimates)
            return model_output
        if self._hp.use_skips:
            embeddings, skips = self.encoder(images)
            skips_flat = []
            for feat in range(0, len(skips), self._hp.skips_stride):
                skips_flat.append(torch.flatten(skips[feat], start_dim=1))
            skips_flat = torch.cat(skips_flat, dim=1)
        else:
            embeddings = self.encoder(images)

        if self._hp.spatial_softmax:
            embeddings = self.spatial_softmax(embeddings)
            embeddings = F.relu(self.linear2(embeddings))
        else:
            embeddings = torch.flatten(embeddings, start_dim=1)
            embeddings = F.relu(self.linear2(embeddings))
            embeddings = F.relu(self.linear3(embeddings))
            embeddings = F.relu(self.linear4(embeddings))
            embeddings = F.relu(self.linear5(embeddings))

        if self._hp.use_skips:
            embeddings = torch.cat((embeddings, skips_flat), dim=1)
            embeddings = F.relu(self.linear_skipmerge(embeddings))

        self.state_estimates = self.linear(embeddings)
        model_output = AttrDict(state_est=self.state_estimates)
        return model_output

    def avg_obj_dist(self, s1, s2):
        distances = []
        for i in range(3):
            dist = (s1[:, i*2:(i+1)*2] - s2[:, i*2:(i+1)*2]).norm(dim=1)
            distances.append(dist)
        mean_batch_dists = torch.mean(torch.stack(distances), dim=0)
        mean_over_batch = torch.mean(mean_batch_dists)
        return mean_over_batch

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        object_loss = self.avg_obj_dist(model_output.state_est.squeeze()[:, 9:], self.labels.to(self._hp.device)[:, 9:])
        self._logger.log_scalar(object_loss, 'avg_obj_dist', step, phase)

    def loss(self, model_output):
        losses = AttrDict()
        setattr(losses, 'mse', torch.nn.MSELoss()(model_output.state_est.squeeze(), self.labels.to(self._hp.device)))
        # compute total loss
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses

    def get_device(self):
        return self._hp.device


def ptrch2uint8(img):
    return ((img + 1)/2*255.).astype(np.uint8)
