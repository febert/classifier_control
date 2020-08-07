from visual_mpc.envs.mujoco_env.base_mujoco_env import BaseMujocoEnv
import numpy as np
import visual_mpc.envs as envs
import copy
from pyquaternion import Quaternion
import os
from gym.spaces import  Dict , Box

from visual_mpc.utils.im_utils import npy_to_mp4
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class TabletopOneObj(BaseMujocoEnv, SawyerXYZEnv):
    """Tabletop Manip (Metaworld) Env"""
    def __init__(self, env_params_dict, reset_state=None):
        hand_low=(-0.2, 0.4, 0.0)
        hand_high=(0.2, 0.8, 0.05)
        obj_low=(-0.3, 0.4, 0.1)
        obj_high=(0.3, 0.8, 0.3)

        dirname = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        params_dict = copy.deepcopy(env_params_dict)
        _hp = self._default_hparams()
        for name, value in params_dict.items():
            print('setting param {} to value {}'.format(name, value))
            _hp.set_hparam(name, value)

        if _hp.textured:
            filename = os.path.join(dirname, "assets/sawyer_xyz/sawyer_oneobj.xml")
        else:
            assert False, "Must use textured mode"
            filename = os.path.join(dirname, "assets/sawyer_xyz/sawyer_multiobject.xml")

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
        default_dict = {
            'verbose': False,
            'difficulty': None,
            'textured': False,
            'render_imgs': True,
        }
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

    def _set_arm_pos_to_start(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[:9] = self._obs_history[0]['qpos'][:9].copy()
        self.set_state(qpos, qvel)
        return self._get_obs()
  #
  # def sample_goal(self):
  #   start_id = 9 + self.targetobj*2
  #   qpos = self.data.qpos.flat.copy()
  #   ogpos = qpos[start_id:(start_id+2)]
  #   goal_pos = np.random.uniform(
  #               -0.3,
  #               0.3,
  #               size=(2,),
  #       )
  #   self._state_goal = goal_pos
  #   self._set_obj_xyz(goal_pos)
  #   self.goalim = self.render()
  #   self._set_obj_xyz(ogpos)
    
    def _reset_hand(self, goal=False):
        pos = self.hand_init_pos.copy()
        for _ in range(10):
          self.data.set_mocap_pos('mocap', pos)
          self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
          self.do_simulation([-1,1], self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM = (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def reset(self, reset_state=None):
        self._reset_hand()

        if reset_state is not None:
            if len(reset_state) == 11:
                target_qpos = reset_state
                target_qvel = np.zeros_like(self.data.qvel)
            else:
                target_qpos = reset_state[:len(reset_state)//2]
                target_qvel = reset_state[len(reset_state) // 2:]
            self.set_state(target_qpos, target_qvel)

        else:
            for i in range(1):
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
        #self.targetobj = np.random.randint(3)
        #self.sample_goal()

        #place = self.targetobj
        #self.curr_path_length = 0
        self._obs_history = []
        o = self._get_obs()
        self._reset_eval()

        #Can try changing this
        return o, self.sim.data.qpos.flat.copy()

    def step(self, action):
        self.set_xyz_action(action[:3])
        #for i in range(100):
        self.do_simulation([action[-1], -action[-1]])
        obs = self._get_obs()
        return obs
  
    def render(self):
        if not self._hp.render_imgs:
            return np.zeros((1, self._hp.viewer_image_height, self._hp.viewer_image_width, 3))
        return super().render().copy()

    def set_goal(self, goal_obj_pose, goal_arm_pose):
        print(f'Setting goals to {goal_obj_pose} and {goal_arm_pose}!')
        super(TabletopOneObj, self).set_goal(goal_obj_pose, goal_arm_pose)

    def get_mean_obj_dist(self):
        distances = self.compute_object_dists(self.sim.data.qpos.flat[9:], self._goal_obj_pose)
        return np.mean(distances)

    def get_distance_score(self):
        """
        :return:  mean of the distances between all objects and goals
        """
        mean_obj_dist = self.get_mean_obj_dist()
        # Pretty sure the below is not quite right...
        arm_dist_despos = np.linalg.norm(self._goal_arm_pose - self.sim.data.qpos[:9])
        print(f'Object distance score is {mean_obj_dist}')
        print(f'Arm joint distance score is {arm_dist_despos}')
        #return arm_dist_despos
        return mean_obj_dist

    def has_goal(self):
        return True

    def compute_object_dists(self, qpos1, qpos2):
        distances = []
        for i in range(1):
            dist = np.linalg.norm(qpos1[i*2:(i+1)*2] - qpos2[i*2:(i+1)*2])
            distances.append(dist)
        return distances

    def goal_reached(self):
        og_pos = self._obs_history[0]['qpos']
        ob_poses = self.sim.data.qpos.flat[9:11]
        #object_dists = self.compute_object_dists(og_pos[9:], ob_poses)
        object_dists = np.linalg.norm(og_pos[9:11]-ob_poses)
        #print('max dist', max(object_dists))
        # enforce that arm moves away
        #gripper_pos = self.get_endeff_pos()[:2]
        #gripper_pos[1] -= 0.6 # do the weird shift
        #print(f'gripper_pos {gripper_pos}')
        #objects = np.array_split(ob_poses, 3)
        #print(f'objects {objects}')
        #object_arm_dists = [np.linalg.norm(gripper_pos - obj) for obj in objects]
        #print(f'obj arm dist {object_arm_dists}')
        #print(f'obj dist {object_dists}')
        return object_dists > 0.075

    def get_jas_at(self, hand_pos):
        self.hand_init_pos = hand_pos
        self._reset_hand()
        return self.sim.data.qpos.flat.copy()[:9]

    def _get_obs(self):
        obs = {}
        #joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:].squeeze())
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:].squeeze())
        obs['gripper'] = self.get_endeff_pos()
        obs['state'] = np.concatenate([copy.deepcopy(self.sim.data.qpos[:].squeeze()),
                                       copy.deepcopy(self.sim.data.qvel[:].squeeze())])
        obs['object_qpos'] = copy.deepcopy(self.sim.data.qpos[9:].squeeze())

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
        return self._get_obs()
  
    def get_goal(self):
        return self.goalim
  
    def has_goal(self):
        return True


   
if __name__ == '__main__':
    env_params = {
      # resolution sufficient for 16x anti-aliasing
      'viewer_image_height': 192,
      'viewer_image_width': 256,
      'textured': True
      #     'difficulty': 'm',
    }
    env = Tabletop(env_params)
    env.reset()
    #
    # import pickle
    # import tqdm
    # poses = []
    # for i in tqdm.tqdm(np.linspace(-0.2, 0.2, num=101)):
    #     for j in np.linspace(0.4, 0.8, num=101):
    #         env.hand_init_pos = np.array((i, j, 0.0))
    #         env.reset()
    #         obs = env._get_obs()
    #         poses.append(obs['qpos'][:9])
    # with open('arm_poses.pkl', 'wb') as f:
    #     pickle.dump(poses, f)

    env.targetobj = 2
    init_pos = np.array([
        0,
        0.2
    ])
    env.obj_init_pos = init_pos
    env._set_obj_xyz(env.obj_init_pos)
    import ipdb; ipdb.set_trace()

    import matplotlib.pyplot as plt
    for i, coord in enumerate(np.linspace(-0.1, 0.1, 21)):
        env.targetobj = 0
        init_pos = np.array([
            coord,
            0.2
        ])
        env.obj_init_pos = init_pos
        env._set_obj_xyz(env.obj_init_pos)
        img = env.render()[0]
        plt.imsave(f'./examples/im_{i}.png', img)
        exit()