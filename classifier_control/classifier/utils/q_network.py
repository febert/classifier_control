import torch
import torch.nn.functional as F
from classifier_control.classifier.utils.layers import Linear
from classifier_control.classifier.utils.subnetworks import ConvEncoder
from classifier_control.classifier.utils.resnet_module import get_resnet_encoder, tile_action_into_image
from torchvision import models


class QNetwork(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        if self._hp.resnet:
            if self._hp.resnet_type == 'resnet50':
                self.resnet = get_resnet_encoder(models.resnet50, 10, channels_in=6+self._hp.action_size, freeze=False)
            elif self._hp.resnet_type == 'resnet34':
                self.resnet = get_resnet_encoder(models.resnet34, 10, channels_in=6 + self._hp.action_size, freeze=False)
            elif self._hp.resnet_type == 'resnet18':
                self.resnet = get_resnet_encoder(models.resnet18, 10, channels_in=6 + self._hp.action_size, freeze=False)
            return
        if self._hp.low_dim:
            self.linear1 = Linear(in_dim=2*self._hp.state_size+self._hp.action_size, out_dim=128, builder=self._hp.builder)
            self.linear2 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
        else:
            self.encoder = ConvEncoder(self._hp)
            out_size = self.encoder.get_output_size()
            self.linear1 = Linear(in_dim=out_size[0]*5*5, out_dim=128, builder=self._hp.builder)
            self.linear2 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
        self.linear3 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
        self.linear4 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
        self.linear5 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
        self.linear6 = Linear(in_dim=128, out_dim=1, builder=self._hp.builder)

    def forward(self, image_pairs, actions):

        if self._hp.resnet:
            actions_tiled = tile_action_into_image(actions, (image_pairs.shape[2], image_pairs.shape[3]))
            resnet_inp = torch.cat((image_pairs, actions_tiled), dim=1)
            return self.resnet(resnet_inp)

        if self._hp.low_dim:
            embeddings = image_pairs
        else:
            embeddings = self.encoder(image_pairs).view(image_pairs.size(0), -1)

        e = F.relu(self.linear1(torch.cat([embeddings, actions], dim=1)))
        e = F.relu(self.linear2(e))
        e = F.relu(self.linear3(e))
        e = F.relu(self.linear4(e))
        e = F.relu(self.linear5(e))

        qvalue = self.linear6(e)  # self.linear6(e)

        return qvalue

  
class DistQNetwork(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        if self._hp.resnet:
            num_channels = 6
            if self._hp.tile_actions:
                num_channels += self._hp.action_size
                outputs = 10
            else:
                self.linear2 = Linear(in_dim=128 + self._hp.action_size, out_dim=128, builder=self._hp.builder)
                self.linear3 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
                self.linear4 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
                self.linear5 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
                self.linear6 = Linear(in_dim=128, out_dim=10, builder=self._hp.builder)
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
            self.linear1 = Linear(in_dim=4, out_dim=128, builder=self._hp.builder)
        else:
            self.encoder = ConvEncoder(self._hp)
            out_size = self.encoder.get_output_size()
            self.linear1 = Linear(in_dim=out_size[0]*5*5, out_dim=128, builder=self._hp.builder)

        self.linear2 = Linear(in_dim=128 + self._hp.action_size, out_dim=128, builder=self._hp.builder)
        self.linear3 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
        self.linear4 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
        self.linear5 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
        self.linear6 = Linear(in_dim=128, out_dim=10, builder=self._hp.builder)

    def forward(self, image_pairs, actions):
        if self._hp.resnet:
            if self._hp.tile_actions:
                actions_tiled = tile_action_into_image(actions, (image_pairs.shape[2], image_pairs.shape[3]))
                resnet_inp = torch.cat((image_pairs, actions_tiled), dim=1)
                out = self.resnet(resnet_inp)
                return F.softmax(out)
            else:
                e = self.resnet(image_pairs)
                e = torch.cat([e, actions], dim=1)
                e = F.relu(self.linear2(e))
                e = F.relu(self.linear3(e))
                e = F.relu(self.linear4(e))
                e = F.relu(self.linear5(e))
                qvalue = F.softmax(self.linear6(e))
                return qvalue

        if self._hp.low_dim:
            embeddings = image_pairs
        else:
            embeddings = self.encoder(image_pairs).view(image_pairs.size(0), -1)
            
        e = F.relu(self.linear1(embeddings))
        e = torch.cat([e, actions], dim=1)
        e = F.relu(self.linear2(e))
        e = F.relu(self.linear3(e))
        e = F.relu(self.linear4(e))
        e = F.relu(self.linear5(e))
        qvalue = F.softmax(self.linear6(e)) #self.linear6(e)
        return qvalue
