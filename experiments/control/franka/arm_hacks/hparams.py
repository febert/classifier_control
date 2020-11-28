""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from classifier_control.environments.sim.cartgripper.cartgripper_xz import CartgripperXZ
from visual_mpc.policy.cem_controllers.samplers.correlated_noise import CorrelatedNoiseSampler

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.envs.robot_envs.base_env import BaseRobotEnv 
from visual_mpc.policy.random.sampler_policy import SamplerPolicy
from classifier_control.cem_controllers.pytorch_classifier_controller import LearnedCostController
from classifier_control.classifier.models.tempdist_regressor import TempdistRegressorTestTime
from classifier_control.environments.sim.tabletop.tabletop_oneobj import TabletopOneObj
from classifier_control.classifier.models.q_function import QFunctionTestTime

env_params = {
    # resolution sufficient for 16x anti-aliasing
    #'email_login_creds': '.email_cred',
    'camera_topics': [IMTopic('/front/image_raw')],
                      # IMTopic('/left/image_raw')],
                      # IMTopic('/right_side/image_raw')],
                      # IMTopic('/left_side/image_raw')],
                      # IMTopic('/right/image_raw')],
    'robot_name':'franka',
    'robot_type':'franka',
    'gripper_attached':'hand',
    'cleanup_rate': -1,
    'duration': 3.5,
    'reopen':False,
    'save_video': True
}


agent = {
    'type': BenchmarkAgent,
    'env': (BaseRobotEnv, env_params),
    'T': 30,
    'gen_xml': (True, 20),  # whether to generate xml, and how often
    # 'make_final_gif_freq':1,
    'start_goal_confs': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/franka_drawer_goals/raw',
    'num_load_steps':30,
}


policy = {
    'type': LearnedCostController,
    'replan_interval': 6,
    'nactions': 13,
    # 'num_samples': 200,
    'selection_frac': 0.05,
    'sampler': CorrelatedNoiseSampler,
    'initial_std':  [0.6, 0.6, 0.3, 0.3], # Can tune this
    'learned_cost': QFunctionTestTime,
    'learned_cost_model_path': os.environ['VMPC_EXP'] + '/classifier_control/distfunc_training/q_function_training/real_drawer_arm_hacks/weights/weights_ep4630.pth' # Point to Q function model weights
    "vidpred_model_path": os.environ['VMPC_EXP'] + '/vpred_models/HDF5TrainableInterface_00000_00000_0_2020-11-26_23-17-56h_d3tcdg/checkpoint_135000',
    'verbose_every_iter': True,
    #'use_gt_model': True,
    'finalweight': -10000,
}

config = {
    'traj_per_file':1,  #28,
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    'save_data': False,
    'save_format': ['raw'],
}

