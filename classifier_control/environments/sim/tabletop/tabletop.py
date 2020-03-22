from visual_mpc.envs.mujoco_env.base_mujoco_env import BaseMujocoEnv
import numpy as np
import visual_mpc.envs as envs
from visual_mpc.envs.mujoco_env.util.create_xml import create_object_xml, create_root_xml, clean_xml
import copy
from pyquaternion import Quaternion
import os
from gym.spaces import  Dict , Box

from visual_mpc.utils.im_utils import npy_to_mp4
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class Tabletop(BaseMujocoEnv, SawyerXYZEnv):
  """Tabletop Manip (Metaworld) Env"""
  def __init__(self, env_params_dict):
    hand_low=(-0.2, 0.4, 0.0)
    hand_high=(0.2, 0.8, 0.05)
    obj_low=(-0.3, 0.4, 0.1)
    obj_high=(0.3, 0.8, 0.3)
    
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "assets/sawyer_xyz/sawyer_multiobject.xml")
    params_dict = copy.deepcopy(env_params_dict)
    _hp = self._default_hparams()
    for name, value in params_dict.items():
      print('setting param {} to value {}'.format(name, value))
      _hp.set_hparam(name, value)
      
    BaseMujocoEnv.__init__(self, filename, _hp)
    
    SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./10,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=filename
        )
    goal_low = self.hand_low
    goal_high = self.hand_high
    self._adim = 4
    self._hp = _hp
    self.liftThresh = 0.04
    self.max_path_length = 100
    self.hand_init_pos = np.array((0, 0.6, 0.0))
    
  def default_ncam():
    return 1

  def _default_hparams(self):
    default_dict = {'verbose':False, 'difficulty': None}
    parent_params = super()._default_hparams()
    for k in default_dict.keys():
      parent_params.add_hparam(k, default_dict[k])
    return parent_params
  
  def _set_obj_xyz(self, pos):
    qpos = self.data.qpos.flat.copy()
    qvel = self.data.qvel.flat.copy()
    start_id = 9 + self.targetobj*2
    qpos[start_id:(start_id+2)] = pos.copy()
    qvel[start_id:(start_id+2)] = 0
    self.set_state(qpos, qvel)
    
  def sample_goal(self):
    start_id = 9 + self.targetobj*2
    qpos = self.data.qpos.flat.copy()
    ogpos = qpos[start_id:(start_id+2)]
    goal_pos = np.random.uniform(
                -0.3,
                0.3,
                size=(2,),
        )
    self._state_goal = goal_pos 
    self._set_obj_xyz(goal_pos) 
    self.goalim = self.render()
    self._set_obj_xyz(ogpos)
    
  def _reset_hand(self, goal=False):
    pos = self.hand_init_pos.copy()
    for _ in range(10):
      self.data.set_mocap_pos('mocap', pos)
      self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
      self.do_simulation([-1,1], self.frame_skip)
    rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
    self.init_fingerCOM  =  (rightFinger + leftFinger)/2
    self.pickCompleted = False

  def get_site_pos(self, siteName):
    _id = self.model.site_names.index(siteName)
    return self.data.site_xpos[_id].copy()
  
  def reset(self, reset_state=None):
    self._reset_hand()
    buffer_dis = 0.04
    block_pos = None
    for i in range(3):
        self.targetobj = i
        init_pos = np.random.uniform(
            -0.2,
            0.2,
            size=(2,),
        )
        self.obj_init_pos = init_pos
        self._set_obj_xyz(self.obj_init_pos)

    for _ in range(100):
      self.do_simulation([0.0, 0.0])
    self.targetobj = np.random.randint(3)
    self.sample_goal()
                           
    place = self.targetobj
    self.curr_path_length = 0
    o = self._get_obs()
        
    #Can try changing this
    return o, None

  def step(self, action):
    self.set_xyz_action(action[:3])
    self.do_simulation([action[-1], -action[-1]])
    obs = self._get_obs()
    return obs
  
  def render(self):
    return super().render().copy()
  
  def _get_obs(self):
    obs = {}
    #joint poisitions and velocities
    obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:].squeeze())
    obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:].squeeze())
    obs['gripper'] = self.get_endeff_pos()

    obs['state'] = np.concatenate([copy.deepcopy(self.sim.data.qpos[:].squeeze()),
                                           copy.deepcopy(self.sim.data.qvel[:].squeeze())])

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
    return self.goalim
  
  def has_goal(self):
    return True

  def goal_reached(self):
    dist = self.get_distance_score()
    if (dist < 0.08):
      return 1
    else:
      return 0
   
  def get_distance_score(self):
    """
        :return:  mean of the distances between all objects and goals
        """
    start_id = 9 + self.targetobj*2
    qpos = self.data.qpos.flat.copy()
    ogpos = qpos[start_id:(start_id+2)]
    dist = np.linalg.norm(ogpos - self._state_goal)
    return dist
