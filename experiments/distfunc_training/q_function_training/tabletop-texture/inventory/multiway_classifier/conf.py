import os
from classifier_control.classifier.utils.general_utils import AttrDict
current_dir = os.path.dirname(os.path.realpath(__file__))
from classifier_control.classifier.models.multiway_tempdist_classifier import MultiwayTempdistClassifer, TesttimeMultiwayTempdistClassifier
from classifier_control.classifier.utils.logger import TdistMultiwayClassifierLogger

configuration = {
    'model': MultiwayTempdistClassifer,
    'model_test': TesttimeMultiwayTempdistClassifier,
    'logger': TdistMultiwayClassifierLogger,
    'data_dir': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/tabletop-texture',       # 'directory containing data.' ,
    'batch_size' : 32,
}

configuration = AttrDict(configuration)

data_config = AttrDict(
                img_sz=(64, 64),
                sel_len=-1,
                T=31)

model_config = {
}
