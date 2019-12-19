from visual_mpc.envs.mujoco_env.base_mujoco_env import BaseMujocoEnv
import numpy as np
import visual_mpc.envs as envs
from visual_mpc.envs.mujoco_env.util.create_xml import create_object_xml, create_root_xml, clean_xml
import copy
from pyquaternion import Quaternion
import os
from visual_mpc.utils.im_utils import npy_to_mp4

class SimpleMaze(BaseMujocoEnv):
  """Simple Maze Navigation Env"""
  def __init__(self, env_params_dict, reset_state = None):
    params_dict = copy.deepcopy(env_params_dict)
    _hp = self._default_hparams()
    for name, value in params_dict.items():
      print('setting param {} to value {}'.format(name, value))
      _hp.set_hparam(name, value)
      
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'simple_maze.xml')
    super().__init__(filename, _hp)
    self._adim = 2
    self.difficulty = _hp.difficulty
    self._hp = _hp
    
  def default_ncam():
    return 1

  def _default_hparams(self):
    default_dict = {'verbose':False, 'difficulty': None}
    parent_params = super()._default_hparams()
    for k in default_dict.keys():
      parent_params.add_hparam(k, default_dict[k])
    return parent_params
  
  def reset(self, reset_state=None):
    if self.difficulty is None:
      self.sim.data.qpos[0] = np.random.uniform(-0.27, 0.27)
    elif self.difficulty == 'e':
      self.sim.data.qpos[0] = np.random.uniform(0.15, 0.27)
    elif self.difficulty == 'm':
      self.sim.data.qpos[0] = np.random.uniform(-0.15, 0.15)
    elif self.difficulty == 'h':
      self.sim.data.qpos[0] = np.random.uniform(-0.27, -0.15)
    self.sim.data.qpos[1] = np.random.uniform(-0.27, 0.27)
    
    self.goal = np.zeros((2,))
    self.goal[0] = np.random.uniform(-0.27, 0.27)
    self.goal[1] = np.random.uniform(-0.27, 0.27)

    # Randomize wal positions
    w1 = np.random.uniform(-0.2, 0.2)
    w2 = np.random.uniform(-0.2, 0.2)
#     print(self.sim.model.geom_pos[:])
#     print(self.sim.model.geom_pos[:].shape)
    self.sim.model.geom_pos[5, 1] = 0.25 + w1
    self.sim.model.geom_pos[7, 1] = -0.25 + w1
    self.sim.model.geom_pos[6, 1] = 0.25 + w2
    self.sim.model.geom_pos[8, 1] = -0.25 + w2
    return self._get_obs(), None

  def step(self, action):
    self.sim.data.qvel[:] = 0
    self.sim.data.ctrl[:] = action
    for _ in range(500):
      self.sim.step()
    obs = self._get_obs()
    self.sim.data.qvel[:] = 0
    return obs
  
  def render(self):
    return super().render().copy()
  
  def _get_obs(self):
    obs = {}
    #joint poisitions and velocities
    obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:].squeeze())
    obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:].squeeze())

    obs['state'] = np.concatenate([copy.deepcopy(self.sim.data.qpos[:self._sdim].squeeze()),
                                           copy.deepcopy(self.sim.data.qvel[:self._sdim].squeeze())])

    #copy non-image data for environment's use (if needed)
    self._last_obs = copy.deepcopy(obs)
    self._obs_history.append(copy.deepcopy(obs))

    #get images
    obs['images'] = self.render()
    obs['env_done'] = False

    return obs
  
  def valid_rollout(self):
    return True

  def current_obs(self):
    return self._get_obs(finger_force)
  
  def get_goal(self):
    curr_qpos = self.sim.data.qpos[:].copy()
    self.sim.data.qpos[:] = self.goal
    self.sim.step()
    goalim = self.render()
    self.sim.data.qpos[:] = curr_qpos
    self.sim.step()
    return goalim
  
  def has_goal(self):
    return True

  def goal_reached(self):
    d = np.sqrt(np.mean((self.goal - self.sim.data.qpos[:])**2))
    if d < 0.05:
      return True
    return False
   
  def get_distance_score(self):
    """
        :return:  mean of the distances between all objects and goals
        """
    d = np.sqrt(np.mean((self.goal - self.sim.data.qpos[:])**2))
    print("********", d)
    if d < 0.1:
      return 1.0
    else:
      return 0.0


  

# def get_model_and_assets():
#   """Returns a tuple containing the model XML string and a dict of assets."""
#   dirname = os.path.dirname(__file__)
#   filename = os.path.join(dirname, 'simple_maze.xml')
#   with open(filename, 'r') as f:
#     data = f.read().replace('\n', '')
#   return data, common.ASSETS


# def navigate(time_limit=_DEFAULT_TIME_LIMIT, random=None,
#              environment_kwargs=None, difficulty=None):
#   """Returns instance of the maze navigation task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = SimpleMaze(random=random, difficulty=difficulty)
#   environment_kwargs = environment_kwargs or {}

#   return control.Environment(
#       physics, task, time_limit=time_limit, control_timestep=0.5,
#       **environment_kwargs)


# class Physics(mujoco.Physics):
#   """Physics simulation."""

#   def tool_to_target(self):
#     """Returns the vector from target to finger in global coordinates."""
#     return (self.named.data.geom_xpos['target', :2] -
#             self.named.data.geom_xpos['toolgeom', :2])

#   def tool_to_target_dist(self):
#     """Returns the signed distance between the finger and target surface."""
#     return np.linalg.norm(self.tool_to_target())


# class SimpleMaze(base.Task):
#   """A Maze Navigation `Task` to reach the target."""

#   def __init__(self, random=None, difficulty=None):
#     """Initialize an instance of `MazeNavigation`.

#     Args:
#       random: Optional, either a `numpy.random.RandomState` instance, an
#         integer seed for creating a new `RandomState`, or None to select a seed
#         automatically (default).
#       difficulty: Optional, a String of 'e', 'm', 'h', for
#       easy, medium or hard difficulty
#     """
#     super(SimpleMaze, self).__init__(random=random)
#     self.difficulty = difficulty
#     self.dontreset = False

#   def initialize_episode(self, physics, difficulty=None):
#     """Sets the state of the environment at the start of each episode."""
#     # Sometime don't reset
#     if self.dontreset:
#       return

#     # Reset based on difficulty
#     if self.difficulty is None:
#       randomizers.randomize_limited_and_rotational_joints(physics, self.random)
#     elif self.difficulty == 'e':
#       physics.data.qpos[0] = self.random.uniform(0.15, 0.27)
#     elif self.difficulty == 'm':
#       physics.data.qpos[0] = self.random.uniform(-0.15, 0.15)
#     elif self.difficulty == 'h':
#       physics.data.qpos[0] = self.random.uniform(-0.27, -0.15)
#     physics.data.qpos[1] = self.random.uniform(-0.27, 0.27)

#     # Randomize wal positions
#     w1 = self.random.uniform(-0.2, 0.2)
#     w2 = self.random.uniform(-0.2, 0.2)
#     physics.named.model.geom_pos['wall1A', 'y'] = 0.25 + w1
#     physics.named.model.geom_pos['wall1B', 'y'] = -0.25 + w1
#     physics.named.model.geom_pos['wall2A', 'y'] = 0.25 + w2
#     physics.named.model.geom_pos['wall2B', 'y'] = -0.25 + w2

#     # Randomize target position
#     physics.named.model.geom_pos['target', 'x'] = self.random.uniform(0.2,
#                                                                       0.28)
#     physics.named.model.geom_pos['target', 'y'] = self.random.uniform(-0.28,
#                                                                       0.28)

#   def get_observation(self, physics):
#     """Returns an observation of the state and positions."""
#     obs = collections.OrderedDict()
#     obs['position'] = physics.position()
#     obs['to_target'] = physics.tool_to_target()
#     obs['velocity'] = physics.velocity()
#     return obs

#   def is_goal(self, physics):
#     """Checks if goal has been reached (within 5 cm)."""
#     d = physics.tool_to_target_dist()
#     if d < 0.05:
#       return True
#     return False

#   def get_reward(self, physics):
#     """Returns shaped reward (not used)."""
#     d = physics.tool_to_target_dist()
#     return -d


