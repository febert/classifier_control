import os
from classifier_control.classifier.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from classifier_control.classifier.models.tempdist_regressor import TempdistRegressor
from classifier_control.classifier.utils.logger import TdistRegressorLogger


configuration = {
    'model': TempdistRegressor,
    'logger': TdistRegressorLogger,
    'batch_size' : 32,
    'dataset': 'robonet',
    'data_dir': '/raid/sudeep/robonet_dataset/v3/hdf5',
    'dataset_conf' : {'RNG': 0, 'ret_fnames': False, 'sub_batch_size': 8, 'action_mismatch': 3,
                     'state_mismatch': 3, 'splits': [0.8, 0.1, 0.1], 'same_cam_across_sub_batch': True,}
}

configuration = AttrDict(configuration)

data_config = AttrDict(
                img_sz=(48, 64),
                sel_len=-1,
                T=31,
                robots=['sawyer'])

model_config = {
}
