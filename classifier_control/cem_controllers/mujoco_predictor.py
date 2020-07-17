import numpy as np
import cv2
import copy
import functools


class MujocoPredictor:

    """
    Use ground truth MuJoCo model as "video predictor"
    """

    def __init__(self, env):
        self.env_class, self.env_params = env[0], copy.deepcopy(env[1])
        self.n_context = 2
        self.horizon = 13
        self.n_workers = 4
        self._input_hparams = {
            'img_size': (64, 64),
        }
        self.env_params['viewer_image_height'] = 48
        self.env_params['viewer_image_width'] = 64

    def get_rand_init(self, n):
        env = self.env_class(self.env_params)
        a = []
        for _ in range(n):
            obs, _ = env.reset()
            a.append(obs['qpos'][:9])
        return np.stack(a)

    def __call__(self, context, actions_dict):
        """
        :param context: dictionary containing context_frames, context_actions, and context_states
        :param actions_dict: dict containing 'actions'
        :return:
        """

        sim_env = self.env_class(self.env_params)
        actions = actions_dict['actions']
        horizon = actions.shape[1]

        curr_state = context['context_states'][-1].squeeze()  # Get current state
        if len(curr_state) == 33:
            curr_qpos, curr_qvel = curr_state[3:3+15], curr_state[3+15:]
        else:
            curr_qpos, curr_qvel = curr_state[:len(curr_state) // 2], curr_state[len(curr_state) // 2:]

        predicted_frames = []
        predicted_states = []
        # For each action sequence in batch
        for b in range(actions.shape[0]):
            batch_ims = []
            batch_states = []
            sim_env.reset(np.concatenate((curr_qpos, curr_qvel)))
            for action in actions[b]:
                obs = sim_env.step(action)
                obs_im = obs['images']
                state = obs['state']
                resized = \
                    (cv2.resize(obs_im.squeeze(), self._input_hparams['img_size'], interpolation=cv2.INTER_AREA) / 255.)[
                        None]
                batch_ims.append(resized)
                batch_states.append(state)
            predicted_frames.append(np.stack(batch_ims))
            predicted_states.append(np.stack(batch_states))
        predicted_frames = np.stack(predicted_frames)
        predicted_states = np.stack(predicted_states)

        del batch_ims, batch_states, sim_env, resized, state, obs_im, obs, actions, context
        print(predicted_states.shape)
        print(predicted_frames.shape)
        return {'predicted_frames': predicted_frames,
                'predicted_states': predicted_states}
