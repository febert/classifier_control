""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from classifier_control.environments.sim.cartgripper.cartgripper_xz import CartgripperXZ
import numpy as np
from visual_mpc.policy.cem_controllers.samplers.correlated_noise import CorrelatedNoiseSampler

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.policy.random.sampler_policy import SamplerPolicy
from classifier_control.cem_controllers.pytorch_classifier_controller import PytorchClassifierController


env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 192,
    'viewer_image_width': 256,
    'cube_objects': True,
    'num_objects': 3,
    'object_object_mindist':0.2,
}


agent = {
    'type': GeneralAgent,
    'env': (CartgripperXZ, env_params),
    'T': 30,
    'gen_xml': (True, 20),  # whether to generate xml, and how often
    'make_final_gif_freq':100
}

classifer_params = {
    'classifier_restore_path': os.environ['VMPC_EXP'] + '/classifier/towel_exp/base_model/model-80000',
}

policy = {
    'type': PytorchClassifierController,
    'replan_interval': 15,
    'num_samples': 600,
    'selection_frac': 0.05,
    'sampler': CorrelatedNoiseSampler,
    'initial_std': [0.05, 0.05],
    'classifier_params': classifer_params,
    'verbose_every_iter': True,
    # "model_params_path": os.environ['VMPC_EXP'] + "/classifier_control/vidpred_training/classifier_control/video_prediction_training/experiment_state-2019-08-13_16-56-42.json",
    # "model_path": os.environ['VMPC_EXP'] + "/classifier_control/vidpred_training/classifier_control/video_prediction_training/HDF5TrainableInterface_0_5324d7c6_2019-08-13_16-56-424fsoe8ps",
    "model_path": '/nfs/kun1/users/febert/data/vmpc_exp/classifier_control/vidpred_training/classifier_control/video_prediction_training/HDF5TrainableInterface_0_5324d7c6_2019-08-13_16-56-424fsoe8ps/checkpoint_30000',
}

config = {
    'traj_per_file':1,  #28,
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 5000,
    'agent': agent,
    'policy': policy,
    'save_format': ['hdf5', 'raw', 'tfrec'],
}
