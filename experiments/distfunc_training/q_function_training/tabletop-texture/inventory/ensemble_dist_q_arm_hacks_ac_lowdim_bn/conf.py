import os
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.logger import TdistClassifierLogger
current_dir = os.path.dirname(os.path.realpath(__file__))
# from experiments.control.sim.multiroom2d import env_benchmark_conf
from classifier_control.classifier.models.ensemble_q import EnsembleQFunction, EnsembleQFunctionTestTime
from classifier_control.classifier.models.dist_q_function import DistQFunction, DistQFunctionTestTime

import imp

configuration = {
    'model': EnsembleQFunction,
    'model_test': EnsembleQFunctionTestTime,
    'logger': TdistClassifierLogger,
    'data_dir': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/tabletop-texture',
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
    #'target_network_update': 'polyak',
    #'polyak': 0.999,
    'q_fn_type': DistQFunction,
    'update_target_rate': 500,
    'sg_sample': 'uniform_distance',
    #'terminal': True,
    'add_arm_hacks': True,
    'ensemble_count': 3,
}
