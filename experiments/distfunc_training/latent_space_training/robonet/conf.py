import os
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.logger import TdistClassifierLogger
current_dir = os.path.dirname(os.path.realpath(__file__))
# from classifier_control.classifier.models.base_tempdistclassifier import BaseTempDistClassifier
from classifier_control.classifier.models.latent_space import LatentSpace
# from experiments.control.sim.multiroom2d import env_benchmark_conf

import imp

configuration = {
    'model': LatentSpace,
    'logger': TdistClassifierLogger,
    'dataset': 'robonet',
    'data_dir': '/raid/sudeep/robonet_dataset/v3/hdf5',
    'dataset_conf' : {'RNG': 0, 'ret_fnames': False, 'sub_batch_size': 8, 'action_mismatch': 3,
                      'state_mismatch': 3, 'splits': [0.8, 0.1, 0.1], 'same_cam_across_sub_batch': True,
                      'epoch_len': 10000}
}

configuration = AttrDict(configuration)

data_config = AttrDict(
                img_sz=(48, 64),
                sel_len=-1,
                T=31,
                robots=['sawyer'])

model_config = {
#     'hidden_size':128,
#     'input_nc': 3,
    'goal_cond': False,
}
