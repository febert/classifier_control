import torch
import torch.nn.functional as F
from classifier_control.classifier.utils.layers import Linear
from classifier_control.classifier.utils.subnetworks import ConvEncoder


class QNetwork(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        if self._hp.low_dim:
            self.linear1 = Linear(in_dim=2*self._hp.state_size, out_dim=128, builder=self._hp.builder)
            self.linear2 = Linear(in_dim=128 + self._hp.action_size, out_dim=128, builder=self._hp.builder)
            self.linear3 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
            self.linear4 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
            self.linear5 = Linear(in_dim=128, out_dim=128, builder=self._hp.builder)
            self.linear6 = Linear(in_dim=128, out_dim=1, builder=self._hp.builder)
        else:
            self.encoder = ConvEncoder(self._hp)
            out_size = self.encoder.get_output_size()
            self.linear1 = Linear(in_dim=64*5*5, out_dim=128, builder=self._hp.builder)
            self.linear2 = Linear(in_dim=128 + self._hp.action_size , out_dim=1, builder=self._hp.builder)

    def forward(self, image_pairs, actions):
        if self._hp.low_dim:
            embeddings = image_pairs
        else:
            embeddings = self.encoder(image_pairs).view(image_pairs.size(0), -1)

        e = F.relu(self.linear1(embeddings))
        e = torch.cat([e, actions], dim=1)
        if self._hp.low_dim:
            e = F.relu(self.linear2(e))
            e = F.relu(self.linear3(e))
            e = F.relu(self.linear4(e))
            e = F.relu(self.linear5(e))
            qvalue = self.linear6(e)
        else:
            qvalue = self.linear2(e)  # self.linear6(e)
        return qvalue
