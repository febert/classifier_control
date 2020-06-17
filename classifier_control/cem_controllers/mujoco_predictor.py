import numpy as np
import cv2
import copy
import functools
from multiprocessing import Pool

def worker(context, input_hp, conf):

    sim_env, actions = conf
    horizon = actions.shape[1]

    curr_state = context['context_states'][-1].squeeze()  # Get current state
    curr_qpos, curr_qval = curr_state[:len(curr_state) // 2], curr_state[len(curr_state) // 2:]

    predicted_frames = []
    predicted_states = []
    # For each action sequence in batch
    for b in range(actions.shape[0]):
        batch_ims = []
        batch_states = []
        sim_env.reset(curr_qpos)

        for action in actions[b]:
            obs = sim_env.step(action)
            obs_im = obs['images']
            state = obs['state']
            resized = \
                (cv2.resize(obs_im.squeeze(), input_hp['img_size'], interpolation=cv2.INTER_AREA) / 255.)[
                    None]
            batch_ims.append(resized)
            batch_states.append(state)
        predicted_frames.append(np.stack(batch_ims))
        predicted_states.append(np.stack(batch_states))
    predicted_frames = np.stack(predicted_frames)
    predicted_states = np.stack(predicted_states)

    return {'predicted_frames': predicted_frames,
            'predicted_states': predicted_states}

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
        self.pool = Pool(self.n_workers)
        self.envs = [self.env_class(self.env_params) for _ in range(self.n_workers)]

    def __call__(self, context, actions_dict):
        """
        :param context: dictionary containing context_frames, context_actions, and context_states
        :param actions_dict: dict containing 'actions'
        :return:
        """
        import time
        tic = time.perf_counter()
        worker_partial = functools.partial(worker, context, self._input_hparams)
        division = np.split(actions_dict['actions'], self.n_workers)
        ans = self.pool.map(worker_partial, zip(self.envs, division))

        predicted_frames = np.concatenate([x['predicted_frames'] for x in ans])
        predicted_states = np.concatenate([x['predicted_states'] for x in ans])

        toc = time.perf_counter()
        print(f"Sim dynamics rollouts elapsed time: {toc-tic:0.4f} seconds")

        return {'predicted_frames': predicted_frames,
                'predicted_states': predicted_states}
