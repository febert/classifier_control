import numpy as np
import cv2
import copy
from multiprocessing import Pool

class MujocoPredictor:

    """
    Use ground truth MuJoCo model as "video predictor"
    """

    def __init__(self, env):
        self.env_class, self.env_params = env[0], copy.deepcopy(env[1])
        self.n_context = 2
        self.horizon = 13
        self._input_hparams = {
            'img_size': (64, 64),
        }
        self.env_params['viewer_image_height'] = 48
        self.env_params['viewer_image_width'] = 64

    def __call__(self, context, actions_dict):
        """
        :param context: dictionary containing context_frames, context_actions, and context_states
        :param actions_dict: dict containing 'actions'
        :return:
        """

        actions = actions_dict['actions'].copy()  # Shape [B, T, a_dim]
        self.horizon = actions.shape[1]
        sim_env = self.env_class(self.env_params)
        curr_state = context['context_states'][-1] # Get current state
        curr_qpos, curr_qval = curr_state[:len(curr_state)//2], curr_state[len(curr_state)//2:]

        predicted_frames = []
        predicted_states = []
        # For each action sequence in batch
        for b in range(actions.shape[0]):
            batch_ims = []
            batch_states = []
            sim_env.reset(curr_qpos)
            #obs_1, _ = sim_env.reset(curr_qpos)
            #obs_im = obs_1['images']
            #resized = \
            #    (cv2.resize(obs_im.squeeze(), self._input_hparams['img_size'], interpolation=cv2.INTER_CUBIC) / 255.)[None]
            #batch_ims.append(resized)
            #batch_states.append(obs_1['state'])
            for action in actions[b]:
                obs = sim_env.step(action)
                obs_im = obs['images']
                state = obs['state']
                resized = (cv2.resize(obs_im.squeeze(), self._input_hparams['img_size'], interpolation=cv2.INTER_AREA)/255.)[None]
                batch_ims.append(resized)
                batch_states.append(state)
            predicted_frames.append(np.stack(batch_ims))
            predicted_states.append(np.stack(batch_states))
        predicted_frames = np.stack(predicted_frames)
        predicted_states = np.stack(predicted_states)

        return {'predicted_frames': predicted_frames,
                'predicted_states': predicted_states}
