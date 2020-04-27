import os
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.logger import TdistClassifierLogger
current_dir = os.path.dirname(os.path.realpath(__file__))
from classifier_control.classifier.models.pu_classifier import PUClassifier, PUClassifierTestTime
# from experiments.control.sim.multiroom2d import env_benchmark_conf

import imp

configuration = {
    'model': PUClassifier,
    'model_test': PUClassifierTestTime,
    'logger': TdistClassifierLogger,
    'data_dir': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/tabletop-texture',       # 'directory containing data.' ,
    'batch_size' : 32,
    # 'num_epochs': 200,
}

configuration = AttrDict(configuration)

data_config = AttrDict(
                img_sz=(64, 64),
                sel_len=-1,
                T=31)

model_config = {
    'pos_priors': [0.22184875, 0.36797679, 0.47712125, 0.56427143, 0.6368221, 0.69897,
                   0.75332767, 0.80163235, 0.84509804, 0.88460658]
}
