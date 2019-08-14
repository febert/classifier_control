from contextlib import contextmanager
import numpy as np
import torch
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
from utils import add_n_dims
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.subnetworks import ConvEncoder
from classifier_control.classifier.utils.spatial_softmax import SpatialSoftmax


from classifier_control.classifier.utils.layers import LayerBuilderParams
from classifier_control.classifier.utils.layers import Linear


class SingleTempDistClassifier(nn.Module):
    def __init__(self, params, tdist, logger):
        super().__init__()
        self._hp = self._default_hparams()
        self.override_defaults(params)  # override defaults with config file
        self.logger = logger
        assert self._hp.batch_size != -1  # make sure that batch size was overridden

        self.tdist = tdist
        self.build_network()
        self._use_pred_length = False

    def _default_hparams(self):
        default_dict = AttrDict({
            'builder':LayerBuilderParams(use_convs=True)
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def build_network(self, build_encoder=True):
        self.encoder = ConvEncoder(self._hp)
        out_size = self.encoder.get_output_size()
        self.spatial_softmax = SpatialSoftmax(out_size[0], out_size[1], out_size[2])
        self.linear = Linear(in_dim=out_size[0], out_dim=2, builder=self._hp.builder)

        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """

        tlen = inputs.demo_seq_images.shape[0]
        pos_pairs, neg_pairs = self.sample_image_pair(inputs.demo_seq_images, tlen, self.tdist)
        image_pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        embeddings = self.encoder(image_pairs)
        embeddings = self.spatial_softmax(embeddings)
        logits = self.linear(embeddings)
        self.softmax_act = self.softmax(logits)
        model_output = AttrDict(logits=logits, pos_pair=self.pos_pair, neg_pair=self.neg_pair)
        return model_output

    def sample_image_pair(self, images, tlen, tdist):

        # get positives:
        t0 = torch.randint(0, tlen - tdist, self._hp.batch_size//2)
        t1 = t0 + torch.randint(0, tdist, self._hp.batch_size//2)
        self.pos_pair = torch.stack([images[t0],images[t1]], dim=1)
        pos_pair_cat = torch.cat([images[t0],images[t1]], dim=-1)

        # get negatives:
        t0 = torch.randint(0, tlen - tdist - 1, self._hp.batch_size//2)
        t1 = torch.randint(t0 + tdist, tlen, self._hp.batch_size//2)
        self.neg_pair = torch.stack([images[t0],images[t1]], dim=1)
        neg_pair_cat = torch.cat([images[t0],images[t1]], dim=-1)

        # one means within range of tdist range,  zero means outside of tdist range
        self.labels = torch.cat([torch.ones(self._hp.batch_size), torch.zeros(self._hp.batch_size)])

        return pos_pair_cat, neg_pair_cat

    def loss(self, inputs, model_output):
        return self.loss(inputs, self.labels)

    # def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
    #     if log_images:
    #         self.logger.log_single_tdist_classifier_image(self.pos_pair, self.neg_pair, self.softmax_act,
    #                                                       'tdist{}'.format(self.tdist), step, phase)