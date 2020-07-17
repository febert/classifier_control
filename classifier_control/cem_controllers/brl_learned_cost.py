import ipdb
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import eval_np
import numpy as np
import os


"""
Wrapper around cost learned using the batch_rl repo 
"""

class BRLLearnedCost:
    def __init__(self, overrideparams, logger=None):
        self._hp = self._default_hparams()
        self._hp.update(overrideparams)

        ptu.set_gpu_mode(True)
        data = torch.load(overrideparams['classifier_restore_path'])
        self.policy = data['evaluation/policy'].stochastic_policy
        self.critic = data['trainer/target_qf1']
        self._hp['goal_cond'] = False

    def _default_hparams(self):
        default_dict = {
            'policy_type': 0,
            'log': False,
        }
        return default_dict

    def eval(self):
        self.critic.eval()
        self.policy.eval()
        return self

    def to(self, device):
        self.critic.to(device)
        self.policy.to(device)
        return self

    def get_device(self):
        # sorry about this hack
        return next(self.critic.parameters()).device

    def get_action(self, inputs):
        state = inputs['current_state']
        goal = inputs['goal_state']
        if self._hp['goal_cond']:
            state = torch.cat((state, goal), dim=1)
        pred_action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value = eval_np(
            self.policy, state, reparameterize=True, deterministic=True, return_log_prob=False)
        return pred_action

    def __call__(self, inputs):
        # state only
        if self._hp['policy_type'] == 0:
            state = inputs['current_state']
            goal = inputs['goal_state']
            if self._hp['goal_cond']:
                state = torch.cat((state, goal), dim=1)
            pred_action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value = eval_np(
                self.policy, state, reparameterize=True, deterministic=True, return_log_prob=False)
            q_value = eval_np(self.critic, state, pred_action)
        # image only
        elif self._hp['policy_type'] == 1:
            pred_action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value = eval_np(
                self.policy, img_x, None, actions=None, reparameterize=True, deterministic=True,
                return_log_prob=False)
        # image + state
        elif self._hp['policy_type'] == 2:
            pred_action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value = eval_np(
                self.policy, img_x, state, actions=None, reparameterize=True, deterministic=True,
                return_log_prob=False)
        else:
            pred_action = np.random.uniform(low=0, high=1, size=(3,))  # random noise
            log_prob = 1
        q_value = q_value.squeeze()
        return -q_value

    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        pass