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

    def get_distance_score(self):

        """
        :return:  mean of the distances between all objects and goals
        """
        mean_obj_dist = super().get_distance_score()

        curr_pos = self.sim.data.qpos[:self._sdim]
        arm_dist_despos = np.linalg.norm(self._goal_arm_pose - curr_pos)

        print('mean_obj_dist', mean_obj_dist)
        print('arm_dist_despos ', arm_dist_despos)
        return (mean_obj_dist + arm_dist_despos)/2


