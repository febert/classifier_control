import torch
import torch.nn.functional as F
import torch.nn as nn
from classifier_control.classifier.utils.layers import Linear
from classifier_control.classifier.utils.subnetworks import ConvEncoder
from classifier_control.classifier.utils.resnet_module import get_resnet_encoder, tile_action_into_image
from torchvision import models
import torchvision
import kornia


class ActorNetwork(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        self.encoder = ConvEncoder(self._hp)
        out_size = self.encoder.get_output_size()
        self.mlp = torch.nn.Sequential()

        self.mlp.add_module('linear_1', Linear(in_dim=out_size[0] * 5 * 5, out_dim=128, builder=self._hp.builder))
        for i in range(3):
            self.mlp.add_module(f'linear_{i+1}', Linear(in_dim=128, out_dim=128, builder=self._hp.builder))
        self.mlp.add_module('linear_final', Linear(in_dim=128, out_dim=self._hp.action_size, builder=self._hp.builder))

    def forward(self, image_pairs):
        embeddings = self.encoder(image_pairs).view(image_pairs.size(0), -1)
        return self.mlp(embeddings)


class QNetwork(torch.nn.Module):
    def __init__(self, hp, distributional=False):
        super().__init__()

        self._hp = hp
        if distributional:
            self.output_activation = F.softmax
            self.num_outputs = self._hp.num_bins
        else:
            self.output_activation = lambda x: x
            self.num_outputs = 1

        self.encoder_output_size = 128

        self.encoder = torch.nn.Sequential()

        if self._hp.resnet:
            num_inp_channels = 6
            self.resnet_normalize = kornia.color.Normalize(mean=torch.FloatTensor([0.485, 0.456, 0.406] * 2),
                                                           std=torch.FloatTensor([0.229, 0.224, 0.225] * 2))
            model_type = None
            if self._hp.resnet_type == 'resnet50':
                model_type = models.resnet50
            elif self._hp.resnet_type == 'resnet34':
                model_type = models.resnet34
            elif self._hp.resnet_type == 'resnet18':
                model_type = models.resnet18
            self.resnet = get_resnet_encoder(model_type, self.encoder_output_size, channels_in=num_inp_channels, freeze=False)

            if self._hp.tile_actions:
                num_inp_channels += self._hp.action_size
            else:
                self.encoder.add_module('linear_1', Linear(in_dim=128 + self._hp.action_size, out_dim=128, builder=self._hp.builder))
                for i in range(3):
                    self.encoder.add_module(f'linear_{i+1}', Linear(in_dim=128, out_dim=128, builder=self._hp.builder))
                self.encoder.add_module('linear_final', Linear(in_dim=128, out_dim=self.encoder_output_size, builder=self._hp.builder))

        elif self._hp.low_dim:
            self.linear1 = Linear(in_dim=2*self._hp.state_size, out_dim=128, builder=self._hp.builder)
        else:
            self.encoder = ConvEncoder(self._hp)
            out_size = self.encoder.get_output_size()
            self.linear1 = Linear(in_dim=out_size[0]*5*5, out_dim=128, builder=self._hp.builder)

        self.linear2 = Linear(in_dim=128 + self._hp.action_size, out_dim=self.encoder_output_size, builder=self._hp.builder)

        self.mlp = torch.nn.Sequential()
        for i in range(3):
            self.mlp.add_module(f'linear_{i}', Linear(in_dim=128, out_dim=128, builder=self._hp.builder))
        self.mlp.add_module('linear_final', Linear(in_dim=128, out_dim=self.num_outputs, builder=self._hp.builder))

    def forward(self, image_pairs, actions):
        if self._hp.resnet:
            # range [-1, 1] to [0, 1]
            image_pairs = self.resnet_normalize((image_pairs / 2.0) + 0.5)
            if self._hp.tile_actions:
                actions_tiled = tile_action_into_image(actions, (image_pairs.shape[2], image_pairs.shape[3]))
                resnet_inp = torch.cat((image_pairs, actions_tiled), dim=1)
                embeddings = self.resnet(resnet_inp)
            else:
                e = self.resnet(image_pairs)
                e = torch.cat([e, actions], dim=1)
                e = self.encoder(e)
                embeddings = e
        else:
            if self._hp.low_dim:
                embeddings = self.encoder(image_pairs)
            else:
                if self._hp.film:
                    inp_dict = {'input': image_pairs, 'act': actions}
                    embeddings = self.encoder(inp_dict).reshape(image_pairs.size(0), -1)
                else:
                    embeddings = self.encoder(image_pairs).view(image_pairs.size(0), -1)

            e = F.relu(self.linear1(embeddings))
            e = torch.cat([e, actions], dim=1)
            embeddings = F.relu(self.linear2(e))

        out = self.mlp(embeddings)
        return self.output_activation(out)


class AugQNetwork(QNetwork):
    def __init__(self, hp, distributional=False):
        super().__init__(hp, distributional)
        self.transform = nn.Sequential(
            nn.ReplicationPad2d(4),
            kornia.augmentation.RandomCrop((self._hp.img_sz[0], self._hp.img_sz[1]))
        )

    def random_crop(self, image_pair):
        if self._hp.crop_goal_ind:
            start_image = self.transform(image_pair[:, :3])
            goal_image = self.transform(image_pair[:, 3:])
            cropped = torch.cat((start_image, goal_image), dim=1)
            return cropped
        else:
            return self.transform(image_pair)

    def forward(self, image_pairs, actions):
        # image_pairs is [B, 6, H, W]
        crops = []
        for _ in range(self._hp.num_crops):
            crop = self.random_crop(image_pairs)
            crops.append(super().forward(crop, actions))
        # [num_crops, B, 6, H, W]
        agg = torch.stack(crops, dim=0)
        return torch.mean(agg, dim=0)
