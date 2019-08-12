from visual_mpc.envs.mujoco_env.cartgripper_env.cartgripper_xz_grasp import CartgripperXZGrasp
import numpy as np

class CartgripperXZ(CartgripperXZGrasp):
    def __init__(self, env_params, reset_state = None):
        super().__init__(env_params, reset_state)
        self._adim, self._sdim = 2, 2      # x z grasp
        self._gripper_dim = None

    def _next_qpos(self, action):
        action = np.concatenate([action, np.array([-1])])  # gripper always open
        return self._previous_target_qpos * self.mode_rel + action

    def has_goal(self):
        return True

    def goal_reached(self):
        obj_pos0 = self._obs_history[0]['object_poses'][0, 0]
        final_obj_pos = self._obs_history[-1]['object_poses'][0, 0]

        print('displacement', np.abs(obj_pos0 - final_obj_pos))
        if np.abs(obj_pos0 - final_obj_pos) > 0.05:
            return True
        else:
            return False
