""" Hyperparameters for Large Scale Data Collection (LSDC) """
import os.path
from visual_mpc.agent.benchmarking_agent import BenchmarkAgent
from classifier_control.environments.sim.cartgripper.cartgripper_xz import CartgripperXZ
from visual_mpc.policy.cem_controllers.samplers.correlated_noise import CorrelatedNoiseSampler

BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

from visual_mpc.policy.random.sampler_policy import SamplerPolicy
from classifier_control.cem_controllers.pytorch_classifier_controller import LearnedCostController
from classifier_control.classifier.models.variants.base_tempdistclassifier_monotonicity import MonotonicityBaseTempDistClassifier


env_params = {
    # resolution sufficient for 16x anti-aliasing
    'viewer_image_height': 192,
    'viewer_image_width': 256,
    'cube_objects': True,
    'num_objects': 1,
    'object_object_mindist':0.2,
}


agent = {
    'type': BenchmarkAgent,
    'env': (CartgripperXZ, env_params),
    'T': 13,
    'gen_xml': (True, 20),  # whether to generate xml, and how often
    # 'make_final_gif_freq':1,
    'start_goal_confs': os.environ['VMPC_DATA'] + '/classifier_control/data_collection/sim/1_obj_cartgripper_xz_startgoal/raw',
    'num_load_steps':30,
}


policy = {
    'type': LearnedCostController,
    'replan_interval': 13,
    'nactions': 13,
    # 'num_samples': 200,
    'selection_frac': 0.05,
    'sampler': CorrelatedNoiseSampler,
    'initial_std': [0.05, 0.05],
    'learned_cost': MonotonicityBaseTempDistClassifier,
    'learned_cost_model_path': os.environ['VMPC_EXP'] + '/classifier_control/distfunc_training/ensem_classifier/cartgripper_xz/weights/weights_ep199.pth',
    'verbose_every_iter': True,
    "vidpred_model_path": os.environ['VMPC_EXP'] + '/classifier_control/vidpred_training/cartgripper_xz/docker_training/HDF5TrainableInterface_0_edf90eee_2019-11-22_04-38-28qp2v6u6q/checkpoint_65000',
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
