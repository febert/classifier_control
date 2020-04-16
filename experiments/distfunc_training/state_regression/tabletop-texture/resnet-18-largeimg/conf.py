import os
from classifier_control.classifier.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from classifier_control.classifier.models.state_regressor import StateRegressor
from classifier_control.classifier.utils.logger import Logger


configuration = {
    'model': StateRegressor,
    'logger': Logger,
    'data_dir': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/tabletop-texture-large',       # 'directory containing data.' ,
    'batch_size': 32,
}

configuration = AttrDict(configuration)

data_config = AttrDict(
                img_sz=(224, 224),
                sel_len=-1,
                T=31)

model_config = {
    'goal_cond': False,
    'state_dim': 15,
    'resnet': True,
    'resnet_type': 'resnet18',
}
