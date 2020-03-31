import torch
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.models.single_tempdistclassifier import SingleTempDistClassifier


class SingleTempDistClassifierMonotone(SingleTempDistClassifier):

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
        if torch.isnan(model_output.logits).any():
            print(f'NaN in logits of {self.tdist}')
            import ipdb; ipdb.set_trace()
        loss = self.cross_ent_loss(model_output.logits, self.labels.to(self._hp.device))
        if torch.isnan(loss).any():
            print(f'NaN in loss of {self.tdist}')
            import ipdb; ipdb.set_trace()
        return loss


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
        fraction = torch.sigmoid(self.linear(embeddings))
        model_output = AttrDict(fraction=fraction)
        return model_output



