from classifier_control.classifier.utils.spatial_softmax import SpatialSoftmax
from classifier_control.classifier.utils.layers import Linear
from classifier_control.classifier.utils.subnetworks import ConvEncoder
from visual_mpc.policy.cem_controllers.visualizer.construct_html import save_imgs_direct
import numpy as np
import torch
from classifier_control.classifier.utils.general_utils import AttrDict
import torch.nn as nn
import torch.nn.functional as F
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.models.single_tempdistclassifier import SingleTempDistClassifier
from classifier_control.classifier.models.utils.utils import select_indices
from classifier_control.classifier.utils.vis_utils import visualize_barplot_array


class MultiwayTempdistClassifer(BaseModel):
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
            'tmax_label':10,  # the highest label for temporal distance, values are clamped after that
            'use_skips':False, #todo try resnet architecture!
            'ngf': 8,
            'nz_enc': 64,
            'classifier_restore_path':None  # not really needed here.
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    @property
    def singletempdistclassifier(self):
        return SingleTempDistClassifier

    def build_network(self, build_encoder=True):
        self.encoder = ConvEncoder(self._hp)
        out_size = self.encoder.get_output_size()
        if self._hp.spatial_softmax:
            self.spatial_softmax = SpatialSoftmax(out_size[1], out_size[2], out_size[0])  # height, width, channel
            self.linear = Linear(in_dim=out_size[0]*2, out_dim=self._hp.tmax_label, builder=self._hp.builder)
        else:
            self.linear = Linear(in_dim=128, out_dim=self._hp.tmax_label, builder=self._hp.builder)
            self.linear2 = Linear(in_dim=out_size[0]*out_size[1]*out_size[2], out_dim=128, builder=self._hp.builder)
            self.linear3 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
            self.linear4 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
            self.linear5 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)

    def sample_image_pair(self, images):
        tlen = images.shape[1]

        # get positives:
        t0 = np.random.randint(0, tlen, self._hp.batch_size)
        t1 = np.array([np.random.randint(t0[b], tlen, 1) for b in range(images.shape[0])]).squeeze()

        if self._hp.use_mixup:
            t0_prime = np.array([np.random.randint(t0[b], t1[b]+1, 1) for b in range(images.shape[0])]).squeeze()
            t0_prime = torch.from_numpy(t0_prime)

        t0, t1 = torch.from_numpy(t0), torch.from_numpy(t1)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)

        self.labels = torch.clamp_max(t1 - t0, self._hp.tmax_label-1)

        if self._hp.use_mixup:
            im_t0_prime = select_indices(images, t0_prime)
            self.labels = self.labels, torch.clamp_max(t1 - t0_prime, self._hp.tmax_label-1)
            im_t0, self.lam = self.mixup_reg(im_t0, im_t0_prime)

        img_pair_stack = torch.stack([im_t0, im_t1], dim=1)
        return img_pair_stack

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        image_pairs = self.sample_image_pair(inputs.demo_seq_images)
        self.img_pair = image_pairs
        model_output = self.make_prediction(image_pairs)
        return model_output

    def make_prediction(self, image_pairs_stacked):
        im_t0, im_t1 = image_pairs_stacked[:,0], image_pairs_stacked[:,1]
        embeddings = self.encoder(torch.cat([im_t0, im_t1], dim=1))
        if self._hp.spatial_softmax:
            embeddings = self.spatial_softmax(embeddings)
        else:
            embeddings = torch.flatten(embeddings, start_dim=1)
            embeddings = F.relu(self.linear2(embeddings))
            embeddings = F.relu(self.linear3(embeddings))
            embeddings = F.relu(self.linear4(embeddings))
            embeddings = F.relu(self.linear5(embeddings))

        logits = self.linear(embeddings)
        self.out_softmax = torch.softmax(logits, dim=1)
        model_output = AttrDict(logits=logits, out_softmax=self.out_softmax, img_pair=image_pairs_stacked)
        return model_output

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        if log_images:
            self._logger.log_pair_predictions(self.img_pair, self.out_softmax, self.labels,'tdist_regression', step, phase)

    def loss(self, model_output):
        losses = AttrDict()
        if self._hp.use_mixup:
            ce_wo_red = torch.nn.CrossEntropyLoss(reduction='none')
            ce_loss = self.lam * ce_wo_red(model_output.logits, self.labels[0].to(self._hp.device)) + \
                      (1-self.lam) * ce_wo_red(model_output.logits, self.labels[1].to(self._hp.device))
            ce_loss = torch.mean(ce_loss)
            self.labels = self.mixup_reg.convex_comb(self.labels[0].cuda(), self.labels[1].cuda(), self.lam)
        else:
            ce_loss = torch.nn.CrossEntropyLoss()(model_output.logits, self.labels.to(self._hp.device))
        setattr(losses, 'cross_entropy', ce_loss)

        # compute total loss
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses

    def get_device(self):
        return self._hp.device


class TesttimeMultiwayTempdistClassifier(MultiwayTempdistClassifer):
    def __init__(self, overrideparams, logger=None, restore_from_disk=True):
        super(TesttimeMultiwayTempdistClassifier, self).__init__(overrideparams, logger)
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

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        image_pairs = torch.stack([inputs['current_img'], inputs['goal_img']], dim=1)
        self.out_softmax = self.make_prediction(image_pairs).out_softmax.data.cpu().numpy().squeeze()
        expected_dist = np.sum((1 + np.arange(self.out_softmax.shape[1])[None]) * self.out_softmax, 1)
        return expected_dist

    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        # save classifier preds
        sel_softmax = self.out_softmax[visualize_indices]
        sigmoid_images = visualize_barplot_array(sel_softmax)
        row_name = 'softmax'
        content_dict[row_name] = save_imgs_direct(verbose_folder,
                                                  row_name, sigmoid_images)

def ptrch2uint8(img):
    return ((img + 1)/2*255.).astype(np.uint8)