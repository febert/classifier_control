from visual_mpc.envs.mujoco_env.base_mujoco_env import BaseMujocoEnv
import numpy as np
import visual_mpc.envs as envs
import copy
from pyquaternion import Quaternion
import os
from gym.spaces import  Dict , Box

from visual_mpc.utils.im_utils import npy_to_mp4
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class FrankaDesk(BaseMujocoEnv, SawyerXYZEnv):
    """Tabletop Manip (Metaworld) Env"""

    def __init__(self, env_params_dict, reset_state=None):
        hand_low = (0.05, -0.75, 0.73)
        hand_high = (0.6, -0.48, 1.0)
        obj_low=(0.15, -0.4, 0.6)
        obj_high=(0.4, -0.6, 0.6)

        dirname = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        params_dict = copy.deepcopy(env_params_dict)
        _hp = self._default_hparams()
        for name, value in params_dict.items():
            print('setting param {} to value {}'.format(name, value))
            _hp[name] = value

        filename = os.path.join(dirname, "./playrooms/rooms/playroom/playroom.xml")

        BaseMujocoEnv.__init__(self, filename, _hp)
        SawyerXYZEnv.__init__(
                self,
                frame_skip=15,
                action_scale=1./5,
                hand_low=hand_low,
                hand_high=hand_high,
                model_name=filename
            )
        goal_low = self.hand_low
        goal_high = self.hand_high
        self.obj_low = obj_low
        self.obj_high = obj_high
        self._adim = 4
        self._hp = _hp
        self.liftThresh = 0.04
        self.max_path_length = 100
        self.hand_init_pos = np.array([0.6, -0.48, 1.0])

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
          parent_params[k] = default_dict[k]
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

    #default_mocap_quat = np.array([0, -1, 0, 1])
    default_mocap_quat = np.array([0, 1, 0.5, 0])

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]

        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', self.default_mocap_quat.copy())
    
    def _reset_hand(self, goal=False):
        pos = self.hand_init_pos.copy()
        pos[0] = np.random.uniform(0.2, self.hand_high[0])
        pos[1] = np.random.uniform(-0.6, self.hand_high[1])
        pos[2] = np.random.uniform(self.hand_low[2], self.hand_high[2])
        for _ in range(20):
          self.data.set_mocap_pos('mocap', pos)
          self.data.set_mocap_quat('mocap', self.default_mocap_quat.copy())
          #self.do_simulation([0]*7 + [-1,1], self.frame_skip)
          self.do_simulation([-1,1], self.frame_skip)
        #rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        #self.init_fingerCOM = (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def get_body_pos(self, bodyName):
        _id = self.model.body_names.index(bodyName)
        return self.data.body_xpos[_id].copy()

    def reset(self, reset_state=None):
        # self.data.qpos[:9] = np.array([5.20311586e-01, 1.07768693e+00, 9.71443297e-01, 7.49235668e-01,
        #        -1.55658590e+00, -8.32050797e-01, 1.96473385e+00, -4.70370014e-07,
        #        1.07759349e-04])
        self._reset_hand()
        if reset_state is not None:
            if reset_state.shape[0] == 41:
                target_qpos = reset_state
                target_qvel = np.zeros_like(self.data.qvel)
            else:
                target_qpos = reset_state[:41]
                target_qvel = reset_state[41:]
            self.set_state(target_qpos, target_qvel)
        else:
            for i in range(3):
                self.targetobj = i
                init_pos = np.random.uniform(
                    self.obj_low,
                    self.obj_high,
                )

                init_pos = [2, 2, 2]
                self.obj_init_pos = init_pos
                self.data.qpos[9+7*i:12+7*i] = init_pos
                #self._set_obj_xyz(self.obj_init_pos)

                self.data.qpos[40] = np.random.uniform(0, 0.3)
                for _ in range(10):
                     #self.do_simulation([0.0, 0.0] + [0]*7)
                     self.do_simulation([0.0, 0.0])
        self.update_mocap_pos()
        self._obs_history = []
        o = self._get_obs()
        self._reset_eval()

        #Can try changing this
        return o, self.sim.data.qpos.flat.copy()
        #return o, None

    def step(self, action):
        self.set_xyz_action(action[:3])
        for i in range(10):
            self.do_simulation([action[-1], -action[-1]])
        self.update_mocap_pos()
        self.do_simulation([action[-1], -action[-1]], 1)
        obs = self._get_obs()
        return obs

    def update_mocap_pos(self):
        # print('mocap', self.data.mocap_pos)
        # print('endeff', self.get_endeff_pos())
        # print(self.data.mocap_pos-self.get_endeff_pos())
        self.data.set_mocap_pos('mocap', self.get_endeff_pos())

    def render(self):
        if not self._hp.render_imgs:
            return np.zeros((1, self._hp.viewer_image_height, self._hp.viewer_image_width, 3))
        return super().render().copy()[..., 25:-25, :]

    def set_goal(self, goal_obj_pose, goal_arm_pose):
        print(f'Setting goals to {goal_obj_pose} and {goal_arm_pose}!')
        super(FrankaDesk, self).set_goal(goal_obj_pose, goal_arm_pose)

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
        for i in range(3):
            dist = np.linalg.norm(qpos1[i*2:(i+1)*2] - qpos2[i*2:(i+1)*2])
            distances.append(dist)
        return distances

    def get_endeff_pos(self):
        #return self.get_site_pos('end_effector').copy()
        return self.get_body_pos('hand').copy()

    def goal_reached(self):
        og_pos = self._obs_history[0]['qpos']
        ob_poses = self.sim.data.qpos.flat[9:15]
        object_dists = self.compute_object_dists(og_pos[9:], ob_poses)
        #print('max dist', max(object_dists))
        # enforce that arm moves away
        gripper_pos = self.get_endeff_pos()[:2]
        gripper_pos[1] -= 0.6 # do the weird shift
        print(f'gripper_pos {gripper_pos}')
        objects = np.array_split(ob_poses, 3)
        print(f'objects {objects}')
        object_arm_dists = [np.linalg.norm(gripper_pos - obj) for obj in objects]
        print(f'obj arm dist {object_arm_dists}')
        print(f'obj dist {object_dists}')
        return np.abs(og_pos[40] - self.sim.data.qpos.flat[40]) > 0.25

    def num_movements(self, dists):
        return np.count_nonzero(np.array(dists) > 0.01)

    def _get_obs(self):
        obs = {}
        #joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:].squeeze())
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:].squeeze())
        obs['gripper'] = self.get_endeff_pos()
        obs['state'] = np.concatenate([obs['gripper'], copy.deepcopy(self.sim.data.qpos[:].squeeze()),
                                         copy.deepcopy(self.sim.data.qvel[:].squeeze())])
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


from moviepy.editor import ImageSequenceClip

def gif(filename, array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

if __name__ == '__main__':

    env_params = {
        # resolution sufficient for 16x anti-aliasing
        'viewer_image_height': 192,
        'viewer_image_width': 256,
        'textured': True,
    }
    env = FrankaDesk(env_params)
    # import mujoco_py
    # rend = mujoco_py.MjViewer(env.sim)
    # for _ in range(10000):
    #     env.sim.step()
    #     rend.render()
    env.reset()
    import pickle
    import cv2
    for i in range(20):
        dir = f'/home/stephentian/vmpc_data/classifier_control/data_collection/sim/franka_desk_highcont_noobj_slide_startgoal/raw/traj_group0/traj{i}/'
        with open(dir + 'obs_dict.pkl', 'rb') as f:
            obs_dict = pickle.load(f)
        state = obs_dict['state'][0]
        state_final = obs_dict['state'][-1]
        state_final[:9] = state[:9]
        env.reset(state_final)
        rend = env.render()[0]
        cv2.imwrite(dir + 'images0/im_30.png', cv2.resize(rend[:, :, ::-1], (64, 64), interpolation=cv2.INTER_AREA))
    # imgs = [env.render()[0]]
    # for i in range(30):
    #     act = [0, 0]
    #     act = np.random.uniform(-1, 1, size=4)
    #     #act = [0, -1, 0, 0]
    #     #act = [-1, 0, 0, 0]
    #     #act = [0] * 4
    #
    #     obs = env.step(act)
    #     rend = env.render()[0]
    #     imgs.append(rend)
    #     cv2.imwrite(f'./test{i}.png', rend)
    #
    # gif('./test.gif', np.stack(imgs))


