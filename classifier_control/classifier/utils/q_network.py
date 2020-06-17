import torch
import torch.nn.functional as F
import torch.nn as nn
from classifier_control.classifier.utils.layers import Linear
from classifier_control.classifier.utils.subnetworks import ConvEncoder
from classifier_control.classifier.utils.resnet_module import get_resnet_encoder, tile_action_into_image
from torchvision import models
import torchvision
import kornia


class QNetwork(torch.nn.Module):
    def __init__(self, hp, num_outputs):
        super().__init__()
        self._hp = hp
        self.num_outputs = num_outputs

        if self._hp.resnet:
            num_channels = 6
            self.resnet_normalize = kornia.color.Normalize(mean=torch.FloatTensor([0.485, 0.456, 0.406] * 2),
                                                           std=torch.FloatTensor([0.229, 0.224, 0.225] * 2))
            if self._hp.tile_actions:
                num_channels += self._hp.action_size
                outputs = self.num_outputs
            else:
                self.linear2 = Linear(in_dim=128 + self._hp.action_size, out_dim=128, builder=self._hp.builder)
                self.linear3 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
                self.linear4 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
                self.linear5 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
                self.linear6 = Linear(in_dim=128, out_dim=1, builder=self._hp.builder)
                outputs = 128
            model_type = None
            if self._hp.resnet_type == 'resnet50':
                model_type = models.resnet50
            elif self._hp.resnet_type == 'resnet34':
                model_type = models.resnet34
            elif self._hp.resnet_type == 'resnet18':
                model_type = models.resnet18
            self.resnet = get_resnet_encoder(model_type, outputs, channels_in=num_channels, freeze=False)
            return

        if self._hp.low_dim:
            self.linear1 = Linear(in_dim=2*self._hp.state_size, out_dim=128, builder=self._hp.builder)
            self.linear2 = Linear(in_dim=128 + self._hp.action_size, out_dim=128, builder=self._hp.builder)
            if self._hp.film:
                self.film1 = Linear(in_dim=self._hp.action_size, out_dim=128, builder=self._hp.builder)
                self.film2 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
                self.film3 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
                self.film4 = Linear(in_dim=128, out_dim=256, builder=self._hp.builder)
        else:
            self.encoder = ConvEncoder(self._hp)
            out_size = self.encoder.get_output_size()
            self.linear1 = Linear(in_dim=out_size[0]*5*5, out_dim=128, builder=self._hp.builder)
            self.linear2 = Linear(in_dim=128 + self._hp.action_size, out_dim=128, builder=self._hp.builder)
        self.linear3 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
        self.linear4 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)

        #self.linear5 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
        #self.linear6 = Linear(in_dim=128, out_dim=1, builder=self._hp.builder)
        self.linear5 = torch.nn.Linear(128, 128, bias=True)
        self.linear6 = torch.nn.Linear(128, 1, bias=True)


    def forward(self, image_pairs, actions):
        if self._hp.resnet:
            # range [-1, 1] to [0, 1]
            image_pairs = self.resnet_normalize((image_pairs / 2.0) + 0.5)
            if self._hp.tile_actions:
                actions_tiled = tile_action_into_image(actions, (image_pairs.shape[2], image_pairs.shape[3]))
                resnet_inp = torch.cat((image_pairs, actions_tiled), dim=1)
                out = self.resnet(resnet_inp)
                return out
            else:
                e = self.resnet(image_pairs)
                e = torch.cat([e, actions], dim=1)
                e = F.relu(self.linear2(e))
                e = F.relu(self.linear3(e))
                e = F.relu(self.linear4(e))
                e = F.relu(self.linear5(e))
                qvalue = self.linear6(e)
                if self.num_outputs > 1:
                    qvalue = F.softmax(qvalue)
                return qvalue

        if self._hp.low_dim:
            embeddings = image_pairs
        else:
            embeddings = self.encoder(image_pairs).reshape(image_pairs.size(0), -1)
        e = F.relu(self.linear1(embeddings))
        if self._hp.film:
            film = F.relu(self.film1(actions))
            film = F.relu(self.film2(film))
            film = F.relu(self.film3(film))
            film = self.film4(film)
            action_gamma, action_beta = film[:, :128], film[:, 128:]
            e = e * action_gamma + action_beta
        else:
            e = torch.cat([e, actions], dim=1)
            e = F.relu(self.linear2(e))
        e = F.relu(self.linear3(e))
        e = F.relu(self.linear4(e))
        e = F.relu(self.linear5(e))
        qvalue = self.linear6(e)  # self.linear6(e)
        if self.num_outputs > 1:
            qvalue = F.softmax(qvalue)
        if self._hp.sigmoid:
            qvalue = F.sigmoid(qvalue)
        return qvalue


class AugQNetwork(QNetwork):
    def __init__(self, hp):
        super().__init__(hp)
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

