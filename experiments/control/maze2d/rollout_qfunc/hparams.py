""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from classifier_control.environments.sim.pointmass_maze.simple_maze import SimpleMaze
from visual_mpc.policy.cem_controllers.samplers.correlated_noise import CorrelatedNoiseSampler

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.policy.random.sampler_policy import SamplerPolicy
from classifier_control.cem_controllers.q_function_controller import QFunctionController
from classifier_control.classifier.models.q_function import QFunctionTestTime


env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 48,
    'viewer_image_width': 64,
    'ncam':1
#     'cube_objects': True,
#     'num_objects': 1,
#     'object_object_mindist':0.2,
}


agent = {
    'type': BenchmarkAgent,
    'env': (SimpleMaze, env_params),
    'T': 50,
    'gen_xml': (True, 20),  # whether to generate xml, and how often
    # 'make_final_gif_freq':1,
    'start_goal_confs': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/maze_navigation_startgoal/raw',
    'num_load_steps':51,
#     'ncam':1
}


policy = {
    'type': QFunctionController,
    'learned_cost_model_path': os.environ['VMPC_EXP'] + '/classifier_control/distfunc_training/dist_q_func_training/maze2d/weights/weights_ep50.pth',
    'verbose_every_iter': True,
}

config = {
    'traj_per_file':1,  #28,
    'current_dir' : current_dir,
    'start_index':0,
    'end_index': 100,
    'agent': agent,
    'policy': policy,
    'save_data': False,
    # 'save_format': ['hdf5', 'raw', 'tfrec'],
}
