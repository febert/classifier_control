from contextlib import contextmanager
import numpy as np
import pdb
import torch
import torch.nn as nn
import kornia
import torch.nn.functional as F
import cv2
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.subnetworks import ConvEncoder, ConvDecoder, FiLM
from classifier_control.classifier.utils.multistep_architecture import Encoder, Decoder

class MultistepDynamics(BaseModel):
    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self.override_defaults(overrideparams)  # override defaults with config file
        self.postprocess_params()

        assert self._hp.batch_size != -1   # make sure that batch size was overridden
        self.build_network()

        if self._hp.random_crops:
            self.transform = nn.Sequential(
                nn.ReplicationPad2d(4),
                kornia.augmentation.RandomCrop((self._hp.img_sz[0], self._hp.img_sz[1]))
            )

    def _default_hparams(self):
        default_dict = AttrDict({
            'use_skips':False, #todo try resnet architecture!
            'action_size': 4,
            'max_t': 10,
            'ngf': 8,
            'nz_enc': 64,
            'lstm_hidden_size': 64,
            'lstm_layers': 1,
            'conv_num_hiddens': 128,
            'conv_num_residual_layers': 3,
            'conv_num_residual_hiddens': 64,
            'latent_size': 128,
            #             'input_nc':3,
            'classifier_restore_path':None,  # not really needed here.,
            'film': False, # don't use film for the conv encoder, but still use it when adding actions
            'fixed_k': 0,
            'fixed_dim_convs': 0,
            'tv_weight': 0.0,
            'random_crops': False,
            'cross_enc_dec_residual': False,
            'geom_sample_p': 0.3,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def build_network(self):
        self.action_LSTM = nn.LSTM(self._hp.action_size, self._hp.lstm_hidden_size, num_layers=self._hp.lstm_layers, bidirectional=True, batch_first=True)
        self.encoder = Encoder(in_channels=3, num_hiddens=self._hp.conv_num_hiddens, num_residual_layers=self._hp.conv_num_residual_layers, num_residual_hiddens=self._hp.conv_num_residual_hiddens)
        out_size = self.encoder.get_output_size()
        out_size = out_size[0] * out_size[1] * out_size[2]
        self.encoder_fc_1 = nn.Sequential(
            nn.Linear(out_size, self._hp.latent_size // 2),
            nn.ReLU(),
            nn.Linear(self._hp.latent_size //2, self._hp.latent_size // 2),
            nn.ReLU(),
            nn.Linear(self._hp.latent_size // 2, self._hp.latent_size // 2),
            nn.ReLU(),
            nn.Linear(self._hp.latent_size // 2, self._hp.latent_size),
            nn.ReLU(),
        )
        self.film_actions = FiLM(self._hp, self._hp.lstm_hidden_size*2, self._hp.latent_size)
        self.film_k = FiLM(self._hp, 1, self._hp.latent_size)
        self.decoder = Decoder(in_channels=self._hp.conv_num_hiddens, num_hiddens=self._hp.conv_num_hiddens,
                               num_residual_layers=self._hp.conv_num_residual_layers,
                               num_residual_hiddens=self._hp.conv_num_residual_hiddens, residual=self._hp.cross_enc_dec_residual)
        self.decoder_fc_1 = nn.Sequential(
            nn.Linear(self._hp.latent_size, self._hp.latent_size // 2),
            nn.ReLU(),
            nn.Linear(self._hp.latent_size // 2, self._hp.latent_size // 2),
            nn.ReLU(),
            nn.Linear(self._hp.latent_size // 2, self._hp.latent_size),
            nn.ReLU(),
            nn.Linear(self._hp.latent_size, out_size),
            nn.ReLU(),
        )

    def build_network_old(self):
        self.action_LSTM = nn.LSTM(self._hp.action_size, self._hp.lstm_hidden_size, num_layers=self._hp.lstm_layers, bidirectional=True, batch_first=True)
        self.encoder = ConvEncoder(self._hp)
        out_size = self.encoder.get_output_size()
        out_size = out_size[0] * out_size[1] * out_size[2]
        self.encoder_fc_1 = nn.Sequential(
            nn.Linear(out_size, self._hp.latent_size),
            nn.ReLU(),
            nn.Linear(self._hp.latent_size, self._hp.latent_size),
            nn.ReLU(),
        )
        self.film_actions = FiLM(self._hp, self._hp.lstm_hidden_size*2, self._hp.latent_size)
        self.film_k = FiLM(self._hp, 1, self._hp.latent_size)
        self.decoder = ConvDecoder(self._hp)
        self.decoder_fc_1 = nn.Sequential(
            nn.Linear(self._hp.latent_size, self._hp.latent_size),
            nn.ReLU(),
            nn.Linear(self._hp.latent_size, out_size),
            nn.ReLU(),
        )

    def predict(self, images, actions, k):
        action_embedding, _ = self.action_LSTM(actions)
        action_embedding = action_embedding[:, -1]
        image_enc, residual = self.encoder(images)
        image_enc = self.encoder_fc_1(image_enc.reshape(image_enc.shape[0], -1))
        image_enc = self.film_actions(image_enc, action_embedding)
        k = torch.log(k.float() + 1)
        image_enc = self.film_k(image_enc, k[:, None])
        image_enc = self.decoder_fc_1(image_enc)
        out_shape = [image_enc.shape[0]] + list(self.encoder.get_output_size())
        if self._hp.cross_enc_dec_residual:
            pred_img = self.decoder(image_enc.view(*out_shape), residual)
        else:
            pred_img = self.decoder(image_enc.view(*out_shape))
        #pred_img = F.sigmoid(pred_img)
        return pred_img

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        tlen = inputs.demo_seq_images.shape[1]
#         print(inputs.demo_seq_images.min(), inputs.demo_seq_images.max(), "*****")
        batch_images = inputs.demo_seq_images
        bs = inputs.demo_seq_images.shape[0]
        if self._hp.fixed_k == 0:
            # get random lengths of k
            k = torch.from_numpy(np.array(np.random.randint(1, self._hp.max_t))).to(self.get_device())
        else:
            # If fixed k is specified, always use it
            k = torch.from_numpy(np.array(self._hp.fixed_k)).to(self.get_device())

        dist = torch.distributions.geometric.Geometric(self._hp.geom_sample_p).sample((bs,)).to(self.get_device()).long()+ 1
        dist = torch.clamp(dist, min=0, max=tlen - k - 1)
        # print(dist)
        # print(k)
        # print(tlen-k.cpu()-dist.cpu())

        t0 = np.random.randint(0, tlen - k.cpu() - dist.cpu(), self._hp.batch_size)
        k = k.repeat(batch_images.shape[0])
        t0 = torch.from_numpy(t0).to(self.get_device())
        batch_actions = inputs.actions
        actions = []
        for i in range(len(t0)):
            actions.append(batch_actions[i, t0[i]:t0[i]+k[i]])
        actions = torch.stack(actions, dim=0)
        start_images = select_indices(batch_images, t0)
        end_images = select_indices(batch_images, t0 + k)
        self.consistency_goals = select_indices(batch_images, t0 + k + dist)
        self.consistency_s_t1= select_indices(batch_images, t0 + k + 1)
        self.consistency_actions = select_indices(batch_actions, t0 + k)
        self.consistency_t0 = t0+k
        self.consistency_t1 = t0+k+1
        self.consistency_tg = t0+k+dist
        if self._hp.random_crops:
            start_images, end_images = self.transform(start_images), self.transform(end_images)
        self.true_images = end_images
        self.pred = self.predict(start_images, actions, k)
        self.start_images = start_images
        self.pairs = torch.stack((self.true_images, (self.pred * 2 - 1)), dim=1)
        self.k = k
        return self.pred, k

    def sample_image_triplet_actions(self, images, actions, tlen, tdist, states):
        
        # get positives:
        t0 = np.random.randint(0, tlen - tdist - 1, self._hp.batch_size)
        t1 = t0 + 1
        tg = t0 + 1 + np.random.randint(0, tdist, self._hp.batch_size)
        t0, t1,  tg = torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(tg)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)
        im_tg = select_indices(images, tg)
        s_t0 = select_indices(states, t0)
        s_t1 = select_indices(states, t1)
        s_tg = select_indices(states, tg)
        pos_act = select_indices(actions, t0)

        self.pos_pair = torch.stack([im_t0, im_tg], dim=1)
        self.pos_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        # get negatives:
        t0 = np.random.randint(0, tlen - tdist - 1, self._hp.batch_size)
        t1 = t0 + 1
        tg = [np.random.randint(t0[b] + tdist + 1, tlen, 1) for b in range(self._hp.batch_size)]
        tg = np.array(tg).squeeze()
        t0, t1, tg = torch.from_numpy(t0), torch.from_numpy(t1), torch.from_numpy(tg)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)
        im_tg = select_indices(images, tg)
        s_t0 = select_indices(states, t0)
        s_t1 = select_indices(states, t1)
        s_tg = select_indices(states, tg)
        neg_act = select_indices(actions, t0)
        self.neg_pair = torch.stack([im_t0, im_tg], dim=1)
        self.neg_pair_cat = torch.cat([im_t0, im_t1, im_tg], dim=1)

        # one means within range of tdist range,  zero means outside of tdist range
        self.labels = torch.cat([torch.ones(self._hp.batch_size), torch.zeros(self._hp.batch_size)])

        return self.pos_pair_cat, self.neg_pair_cat, pos_act, neg_act

    def total_variation_loss(self, img, weight):
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

    def loss(self, model_output):

        BCE = ((self.pred - ((self.true_images + 1) / 2.0))**2).mean()
        #BCE = torch.nn.L1Loss()(self.pred, ((self.true_images + 1) / 2.0))
        # for i in range(10):
        #     rec = self.rec[i, 25].permute(1,2,0).cpu().detach().numpy() * 255.0
        #     im = ((self.images + 1 ) / 2.0)[i, 25].permute(1,2,0).cpu().detach().numpy() * 255.0
        #     ex = np.concatenate([rec,im], 0)
        #     cv2.imwrite("ex"+str(i)+".png", ex)
        #
#         print(BCE)
        #KLD = -0.5 * torch.mean(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
#         print(KLD)
        losses = AttrDict()
        losses.reconstruction = BCE
        losses.tv_loss = self.total_variation_loss(self.pred, self._hp.tv_weight)
        #losses.total_loss = losses
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses
    
    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        if log_images:
            self._logger.log_predictions(self.start_images, self.pairs, self.k.squeeze(), self.k.squeeze(),
                                                          'multistep_preds', step, phase)

    def get_device(self):
        return self._hp.device
    
def select_indices(tensor, indices):
    new_images = []
    for b in range(tensor.shape[0]):
        new_images.append(tensor[b, indices[b]])
    tensor = torch.stack(new_images, dim=0)
    return tensor

class MultistepDynamicsTestTime(MultistepDynamics):
    def __init__(self, overrideparams, logger=None):
        super(MultistepDynamicsTestTime, self).__init__(overrideparams, logger)
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
