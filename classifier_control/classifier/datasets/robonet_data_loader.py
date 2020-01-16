from robonet.datasets.robonet_dataset import RoboNetDataset

class RoboNetVideoDataset(RoboNetDataset):
    # TODO: Name this class better
    # This is a wrapper around the RoboNetDataset torch loader class
    def __init__(self, dataset_files_or_metadata, mode='train', hparams=dict()):
        super(RoboNetVideoDataset, self).__init__(dataset_files_or_metadata, mode, hparams)

    def _postprocess_fn(self, data):
        return self.process_images(super(RoboNetVideoDataset, self)._postprocess_fn(data))

    @staticmethod
    def process_images(dict_inp):
        images = dict_inp['images']
        if len(images.shape) == 5:
            images = images[:, 0]  # Number of cameras, used in RL environments
        images = images * 2 - 1
        dict_inp['demo_seq_images'] = images
        return dict_inp


