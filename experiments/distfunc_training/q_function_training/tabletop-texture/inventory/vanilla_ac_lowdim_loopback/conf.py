import os
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.logger import TdistClassifierLogger
current_dir = os.path.dirname(os.path.realpath(__file__))
from classifier_control.classifier.models.q_function import QFunction, QFunctionTestTime
# from experiments.control.sim.multiroom2d import env_benchmark_conf

import imp

configuration = {
    'model': QFunction,
    'model_test': QFunctionTestTime,
    'logger': TdistClassifierLogger,
    'data_dir': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/tabletop-texture-loopback',
    'batch_size': 32,
    'num_epochs': 1000,
}

configuration = AttrDict(configuration)

data_config = AttrDict(
                img_sz=(64, 64),
                sel_len=-1,
                T=31)

model_config = {
    'low_dim':True,
    'gamma':0.8,
    'action_size': 4,
    'optimize_actions': 'actor_critic',
    'update_target_rate': 1000,
    #'terminal': True,
    'normalization': 'none',
    'use_batchnorm': False,
}
