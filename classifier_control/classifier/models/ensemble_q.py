import numpy as np
import torch
import torch.nn.functional as F
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.utils.actor_network import ActorNetwork
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.utils.q_network import QNetwork, AugQNetwork
from classifier_control.classifier.models.q_function import QFunction
from classifier_control.classifier.models.dist_q_function import DistQFunction
from torch.optim import Adam
import torch.nn as nn


class EnsembleQFunction(QFunction):
    def __init__(self, overrideparams, logger=None):
        BaseModel.__init__(self, logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self.override_defaults(overrideparams)  # override defaults with config file
        self.postprocess_params()

        assert self._hp.batch_size != -1   # make sure that batch size was overridden
        q_params = self.overrideparams.copy()
        q_params.pop('ensemble_count')
        q_fn_type = q_params.pop('q_fn_type') if 'q_fn_type' in q_params else QFunction
        self.q_functions = nn.ModuleList([q_fn_type(q_params, logger) for _ in range(self._hp.ensemble_count)])
        self.proxy_ctrl_counter = 0
        self.q_fn_type = q_fn_type
        self.hm_counter = 0

    def init_optimizers(self, hp):
        for q_fn in self.q_functions:
            q_fn.init_optimizers(hp)
        self.optimizer = self.q_functions[0].optimizer

    def optim_step(self, output):
        loss_dicts = []
        for model_output, model in zip(output['model_outputs'], self.q_functions):
            loss_dicts.append(model.optim_step(model_output))
        return self._reorg_loss_dict(loss_dicts)

    def _reorg_loss_dict(self, d):
        loss_dict = AttrDict()
        total_losses = []
        for i, indiv_losses in enumerate(d):
            for key in indiv_losses:
                loss_dict[f'model_{i}_{key}'] = indiv_losses[key]
                if key == 'total_loss':
                    total_losses.append(indiv_losses[key])
        loss_dict.total_loss = torch.stack(total_losses).sum(dim=0)
        return loss_dict

    def loss(self, model_output):
        loss_dicts = []
        for model_output, model in zip(model_output['model_outputs'], self.q_functions):
            loss_dicts.append(model.loss(model_output))
        return self._reorg_loss_dict(loss_dicts)

    def _default_hparams(self):
        default_dict = AttrDict({
            'ensemble_count': 1,
            'q_fn_type': QFunction,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def train(self, mode=True):
        for q_fn in self.q_functions:
            q_fn.train(mode)
        return self

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        #### Train vs Test
        model_outputs = [f(inputs) for f in self.q_functions]
        if self.q_fn_type == DistQFunction:
            qval = np.stack(model_outputs).max(axis=0)
        else:
            qval = np.stack(model_outputs).min(axis=0)
        if "demo_seq_images" in inputs.keys():
            return {
                'qval': qval,
                'model_outputs': model_outputs
            }
        else:
            return qval, model_outputs

    def get_max_q(self, image_pairs):
        """
        :param image_pairs: image pairs (s)
        :return: max_a Q(s, a)
        """
        x = [f.get_max_q(image_pairs) for f in self.q_functions]
        return torch.stack(x).max(dim=0)

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase, prefix=''):

        if log_images and self.proxy_ctrl_counter == 50 and self._hp.log_control_proxy:
            self.proxy_ctrl_counter = 0
            # Compute proxy control perf
            self.log_control_proxy(lambda inp: -1 * self(inp), step, phase)

        if log_images and self._hp.low_dim and self.hm_counter == 20:
            self.hm_counter = 0
            # Log heatmaps
            heatmaps_vary_curr = []
            heatmaps_vary_goal = []
            x_range = -0.3, 0.3
            y_range = -0.3, 0.3
            for i in range(5):
                if i % 2 == 0:
                    hm_object_center = torch.FloatTensor((0.1, 0.1))
                else:
                    hm_object_center = torch.FloatTensor((0.0, 0.0))
                state = self.q_functions[0].images[i, :self._hp.state_size]
                if self._hp.state_size == 22:
                    vary_ind = 0
                else:
                    vary_ind = np.random.randint(0, 3)  # index of object to move
                goal_state = state.clone()
                goal_state[9 + 2 * vary_ind:11 + 2 * vary_ind] = hm_object_center
                goal_state_rep = goal_state[None].repeat(101 * 101, 1)
                xy_range = torch.linspace(x_range[0], x_range[1], steps=101).cuda()
                outer_prod = torch.stack((xy_range.repeat(101), torch.repeat_interleave(xy_range, repeats=101, dim=0)),
                                         dim=1)
                curr_states = torch.cat(
                    (goal_state_rep[:, :9 + 2 * vary_ind], outer_prod, goal_state_rep[:, 11 + 2 * vary_ind:]), dim=1)

                image_pairs = torch.cat((curr_states, goal_state_rep), dim=1)
                image_pairs_flip = torch.cat((goal_state_rep, curr_states), dim=1)
                max_qs = self.get_max_q(image_pairs).detach().cpu().numpy()
                max_qs_flip = self.get_max_q(image_pairs_flip).detach().cpu().numpy()
                heatmaps_vary_curr.append(self.get_heatmap(max_qs, x_range, y_range))
                heatmaps_vary_goal.append(self.get_heatmap(max_qs_flip, x_range, y_range))
                del curr_states, image_pairs, max_qs, outer_prod
                torch.cuda.empty_cache()
            heatmaps_vary_curr = np.stack(heatmaps_vary_curr)
            heatmaps_vary_goal = np.stack(heatmaps_vary_goal)
            self._logger.log_images(heatmaps_vary_curr, f'{prefix}heatmaps_objects_vary_curr', step, phase)
            self._logger.log_images(heatmaps_vary_goal, f'{prefix}heatmaps_objects_vary_goal', step, phase)

            heatmaps_vary_curr = []
            heatmaps_vary_goal = []
            x_range = -0.2, 0.2
            y_range = 0.4, 0.8
            import pickle as pkl
            import pathlib
            path = pathlib.Path(__file__).parent.absolute()
            with open(path.joinpath('arm_poses.pkl'), 'rb') as f:
                arm_poses = pkl.load(f)
            arm_poses = torch.FloatTensor(arm_poses).cuda()
            for i in range(5):
                state = self.images[i, :self._hp.state_size]
                goal_state = state.clone()
                goal_state_rep = goal_state[None].repeat(101 * 101, 1)
                curr_states = torch.cat(
                    (arm_poses, goal_state_rep[:, 9:]), dim=1)

                image_pairs = torch.cat((curr_states, goal_state_rep), dim=1)
                image_pairs_flip = torch.cat((goal_state_rep, curr_states), dim=1)
                max_qs = self.get_max_q(image_pairs).detach().cpu().numpy()
                max_qs_flip = self.get_max_q(image_pairs_flip).detach().cpu().numpy()
                heatmaps_vary_curr.append(self.get_heatmap(max_qs, x_range, y_range))
                heatmaps_vary_goal.append(self.get_heatmap(max_qs_flip, x_range, y_range))

                del curr_states, image_pairs, max_qs
                torch.cuda.empty_cache()
            heatmaps_vary_curr = np.stack(heatmaps_vary_curr)
            heatmaps_vary_goal = np.stack(heatmaps_vary_goal)
            self._logger.log_images(heatmaps_vary_curr, f'{prefix}heatmaps_arm_vary_curr', step, phase)
            self._logger.log_images(heatmaps_vary_goal, f'{prefix}heatmaps_arm_vary_goal', step, phase)

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        # Log generally useful outputs
        self._log_losses(losses, step, phase)

        if phase == 'train':
            self.log_gradients(step, phase)

        for i, model_out, q_fn in zip(range(len(self.q_functions)), model_output['model_outputs'], self.q_functions):
            q_fn._log_outputs(model_out, inputs, losses, step, log_images, phase, prefix=f'model_{i}_')
        self._log_outputs(model_output, inputs, losses, step, log_images, phase)


class EnsembleQFunctionTestTime(EnsembleQFunction):
    def __init__(self, overrideparams, logger=None):
        super(EnsembleQFunctionTestTime, self).__init__(overrideparams, logger)
        if self._hp.classifier_restore_path is not None:
            checkpoint = torch.load(self._hp.classifier_restore_path, map_location=self._hp.device)
            self.load_state_dict(checkpoint['state_dict'])
            for q_fn in self.q_functions:
                q_fn.target_qnetwork.load_state_dict(q_fn.qnetwork.state_dict())
                q_fn.target_qnetwork.eval()
                if self.actor_critic:
                    q_fn.target_pi_net.load_state_dict(q_fn.pi_net.state_dict())
                    q_fn.target_pi_net.eval()
        else:
            print('#########################')
            print("Warning Q function weights not restored during init!!")
            print('#########################')

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('classifier_restore_path', None)
        return parent_params
      
    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        pass
      
    def forward(self, inputs):
        lamb = 1
        qvals, model_outs = super().forward(inputs)
        #return -qvals + lamb * np.std(np.stack(model_outs), axis=0)
        if self.q_fn_type == DistQFunction:
            return qvals
        else:
            return -qvals

