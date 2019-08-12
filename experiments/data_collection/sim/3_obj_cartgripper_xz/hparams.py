""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from environments.sim.cartgripper.cartgripper_xz import CartgripperXZ
import numpy as np

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.policy.random.sampler_policy import SamplerPolicy


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

policy = {
    'type' : SamplerPolicy,
    'nactions': 15,
    'initial_std':  [0.05, 0.05],
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
