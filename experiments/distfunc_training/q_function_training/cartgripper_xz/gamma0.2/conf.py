import os
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.logger import TdistClassifierLogger
current_dir = os.path.dirname(os.path.realpath(__file__))
# from classifier_control.classifier.models.base_tempdistclassifier import BaseTempDistClassifier
from classifier_control.classifier.models.q_function import QFunction
# from experiments.control.sim.multiroom2d import env_benchmark_conf

import imp

configuration = {
    'model': QFunction,
    'logger': TdistClassifierLogger,
    'data_dir': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/1_obj_cartgripper_xz_rejsamp',       # 'directory containing data.' ,
    'batch_size' : 32,
    # 'num_epochs': 200,
}

configuration = AttrDict(configuration)

data_config = AttrDict(
                img_sz=(64, 64),
                sel_len=-1,
                T=31)

model_config = {
#     'low_dim':True,
    'gamma':0.2
}
