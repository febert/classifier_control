from visual_mpc.policy.policy import Policy
import imp
import numpy as np
from visual_mpc.policy.cem_controllers.visualizer.construct_html import save_gifs, save_html, save_img, fill_template, img_entry_html, save_imgs, save_gifs_direct, save_imgs_direct, save_img_direct, save_img, save_html_direct
from visual_mpc.video_prediction.pred_util import get_context, rollout_predictions
from collections import OrderedDict
from classifier_control.classifier.utils.DistFuncEvaluation import DistFuncEvaluation
from classifier_control.classifier.models.base_tempdistclassifier import BaseTempDistClassifierTestTime
from classifier_control.classifier.models.dist_q_function import DistQFunctionTestTime
from classifier_control.classifier.models.q_function import QFunctionTestTime
from classifier_control.classifier.models.dist_q_function import DistQFunctionTestTime
from classifier_control.classifier.models.gc_bc import GCBCTestTime
from classifier_control.classifier.datasets.data_loader import FixLenVideoDataset

from robonet.video_prediction.testing import VPredEvaluation
import torch
import os
import cv2
import networkx as nx
import tqdm

LOG_SHIFT = 1e-5

def resample_imgs(images, img_size):
    if images.shape[-1] > img_size[-1]:
        interp_type = cv2.INTER_AREA
    else:
        interp_type = cv2.INTER_CUBIC
    if len(images.shape) == 5:
        resized_images = np.zeros([images.shape[0], 1, img_size[0], img_size[1], 3], dtype=np.uint8)
        for t in range(images.shape[0]):
            resized_images[t] = \
            cv2.resize(images[t].squeeze(), (img_size[1], img_size[0]), interpolation=interp_type)[None]
        return resized_images
    elif len(images.shape) == 3:
        return cv2.resize(images, (img_size[1], img_size[0]), interpolation=interp_type)

class SORBController(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """

        :param ag_params: agent parameters
        :param policyparams: policy parameters
        :param gpu_id: starting gpu id
        :param ngpu: number of gpus
        """
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.agentparams = ag_params
        self.img_sz = (64, 64)
        learned_cost_testparams = {}
        learned_cost_testparams['batch_size'] = self._hp.num_samples
        learned_cost_testparams['data_conf'] = {'img_sz': self.img_sz}  #todo currently uses 64x64!!
        learned_cost_testparams['classifier_restore_path'] = self._hp.learned_cost_model_path
        learned_cost_testparams['classifier_restore_paths'] = ['']

        self.learned_cost = DistFuncEvaluation(DistQFunctionTestTime, learned_cost_testparams)
        self.device = self.learned_cost.model.get_device()
        learned_cost_dir = os.path.dirname(learned_cost_testparams['classifier_restore_path'])
        graph_dir = learned_cost_dir + '/graph.pkl'
        if not os.path.isfile(graph_dir):
            self.preconstruct_graph(graph_dir)

        self.graph, self.graph_states = self.construct_graph(graph_dir)

        inv_model_testparams = {}
        inv_model_testparams['batch_size'] = self._hp.num_samples
        inv_model_testparams['data_conf'] = {'img_sz': self.img_sz}  # todo currently uses 64x64!!
        inv_model_testparams['classifier_restore_path'] = self._hp.inv_model_path
        inv_model_testparams['classifier_restore_paths'] = ['']

        self.inverse_model = DistFuncEvaluation(GCBCTestTime, inv_model_testparams)

        self._img_height, self._img_width = [ag_params['image_height'], ag_params['image_width']]

        self._adim = self.agentparams['adim']
        self._sdim = self.agentparams['sdim']

        self._n_cam = 1 #self.predictor.n_cam

        self._desig_pix = None
        self._goal_pix = None
        self._images = None

        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None

    def compute_pairwise_dist(self, v1, v2=None):
        if v2 is None:
            v2 = v1
        dists = []
        if not torch.is_tensor(v2):
            v2 = torch.FloatTensor(v2)
        if not torch.is_tensor(v1):
            v1 = torch.FloatTensor(v1)

        if v2.shape[0] == 1:
            curr = 0
            while curr < v1.shape[0]:
                batch = v1[curr:curr+self._hp.dloader_bs]
                inp_dict = {'current_img': batch.cuda(),
                            'goal_img': v2.repeat(batch.shape[0], 1, 1, 1).cuda(), }
                score = self.learned_cost.predict(inp_dict)
                if not hasattr(score, '__len__'):
                    score = np.array([score])
                dists.append(score)
                curr += self._hp.dloader_bs
            dists = np.concatenate(dists)[None]
        else:
            for i, image in tqdm.tqdm(enumerate(v1)):
                inp_dict = {'current_img': image[None].repeat(v2.shape[0], 1, 1, 1).cuda(),
                            'goal_img': v2.cuda(),}
                score = self.learned_cost.predict(inp_dict)
                dists.append(score)
        dists = np.stack(dists)
        # dists = -dists
        # dists = np.clip(dists, 5+1e-5, 10)
        # # Convert to tdist
        # gamma = 0.8
        # dists = np.log((dists * (1-gamma) - 1) / (9-10*gamma)) / np.log(0.8) + 1
        return dists

    def preconstruct_graph(self, cache_fname):
        images = self.get_random_observations()
        dist = self.compute_pairwise_dist(images)
        graph = {'images': images.cpu().numpy(), 'dists': dist}
        import pickle as pkl
        with open(cache_fname, 'wb') as f:
            pkl.dump(graph, f)

    def construct_graph(self, cache_fname):
        # Load cache
        import pickle as pkl
        with open(cache_fname, 'rb') as f:
            data = pkl.load(f)
            images, dists = data['images'], data['dists']
        g = nx.DiGraph()
        for i, s_i in enumerate(images):
            for j, s_j in enumerate(images):
                length = dists[i, j]
                if self.dist_check(length):
                    g.add_edge(i, j, weight=length)
        return g, images

    def get_random_observations(self):
        from attrdict import AttrDict
        hp = AttrDict(img_sz=(64, 64),
                      sel_len=-1,
                      T=31)
        dataset = FixLenVideoDataset(self._hp.graph_dataset, self.learned_cost.model._hp, hp).get_data_loader(self._hp.dloader_bs)
        total_images = []
        dl = iter(dataset)
        for i in range(self._hp.graph_size // self._hp.dloader_bs):
            try:
                batch = next(dl)
            except StopIteration:
                dl = iter(dataset)
                batch = next(dl)
            images = batch['demo_seq_images']
            selected_images = images[torch.arange(len(images)), torch.randint(0, images.shape[1], (len(images),))]
            total_images.append(selected_images)
        total_images = torch.cat(total_images)
        return total_images

    def reset(self):
        self._expert_score = None
        self._images = None
        self._expert_images = None
        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None
        return super(SORBController, self).reset()

    def _default_hparams(self):
        default_dict = {
            'learned_cost_model_path': None,
            'inv_model_path': None,
            'verbose_every_iter': False,
            'dist_q': True,
            'graph_dataset': None,
            'graph_size': 5000,
            'dloader_bs': 500,
            'num_samples': 200,
            'max_dist': 15.0,
            'min_dist': 0.0,
        }
        parent_params = super(SORBController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params[k] = default_dict[k]
        return parent_params

    def dist_check(self, dist):
        return self._hp.min_dist < dist < self._hp.max_dist

    def get_waypoint(self, input_images, goal_img):
        g2 = self.graph.copy()
        start_to_rb = self.compute_pairwise_dist(input_images[None], self.graph_states).flatten()
        rb_to_goal = self.compute_pairwise_dist(self.graph_states, goal_img).flatten()
        start_to_goal = self.compute_pairwise_dist(input_images[None], goal_img).flatten().squeeze()
        for i, (dist_from_start, dist_to_goal) in enumerate(zip(start_to_rb, rb_to_goal)):
            if self.dist_check(dist_from_start):
                g2.add_edge('start', i, weight=dist_from_start)
            if self.dist_check(dist_to_goal):
                g2.add_edge(i, 'goal', weight=dist_to_goal)
        try:
            path = nx.shortest_path(g2, 'start', 'goal', weight='weight')
            edge_lengths = []
            for (i, j) in zip(path[:-1], path[1:]):
                edge_lengths.append(g2[i][j]['weight'])
        except:
            path = ['start', 'goal']
            edge_lengths = [start_to_goal]

        wypt_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]  # Reverse CumSum
        print(path)
        waypoint_vec = list(path)[1:-1]
        print(wypt_to_goal_dist)
        verbose_folder = self.traj_log_dir
        plan_imgs = [self.graph_states[i] for i in waypoint_vec]
        plan_imgs_cat = np.concatenate([input_images.cpu().numpy()] + plan_imgs + [goal_img[0].cpu().numpy()], axis=1)
        plan_imgs_cat = np.transpose((plan_imgs_cat + 1) / 2 * 255, [1, 2, 0])
        cv2.imwrite(verbose_folder + '/plan_{}.png'.format(self._t), plan_imgs_cat[:, :, ::-1])

        return waypoint_vec, wypt_to_goal_dist[1:], edge_lengths[0], start_to_goal

    def get_best_action(self, t=None):
        resampled_imgs = resample_imgs(self._images, self.img_sz) / 255.
        input_images = ten2pytrch(resampled_imgs, self.device)[-1]
        goal_img = uint2pytorch(resample_imgs(self._goal_image, self.img_sz), self._hp.num_samples, self.device)

        waypoints, graph_dists, first_wp_dist, start_to_goal = self.get_waypoint(input_images, goal_img[0][None])
        print(first_wp_dist)
        if len(waypoints) > 0 and (first_wp_dist < start_to_goal or start_to_goal > self._hp.max_dist):
            print('using waypoint')
            wpt_goal = torch.FloatTensor(self.graph_states[waypoints[0]])[None].cuda()
        else:
            print('using original goal')
            wpt_goal = goal_img[0][None]
        #act = self.learned_cost.model.target_pi_net(torch.cat((input_images[None], wpt_goal), dim=1))[0].cpu().detach().numpy()
        #print('target pi net act', act)

        inp_dict = {
            'current_img': input_images[None],
            'goal_img': wpt_goal,
        }

        act = self.inverse_model.predict(inp_dict).action[0].cpu().detach().numpy()
        print('GCBC act', act)

        # cem_itrs = 3
        # mean = np.zeros(self._adim)
        # std = np.ones(self._adim) * 0.6
        # for i in range(3):
        #     try_actions = np.random.normal(mean, std, size=(self._hp.num_samples, self._adim))
        #     try_actions = np.clip(try_actions, -1, 1)
        #     #try_actions[0] = act
        #     wpt_goal_repeated = wpt_goal.repeat(self._hp.num_samples, 1, 1, 1)
        #     try_actions_tensor = torch.FloatTensor(try_actions).cuda()
        #     inp_dict = {
        #         'current_img' : input_images[None].repeat(self._hp.num_samples, 1, 1, 1),
        #         'goal_img': wpt_goal_repeated,
        #         'actions': try_actions_tensor
        #     }
        #
        #     qvalues = self.learned_cost.predict(inp_dict)
        #     best_action_ind = np.argmin(qvalues, axis=0)
        #     act = try_actions[best_action_ind]
        #
        #     inds = np.argsort(qvalues)[:int(self._hp.num_samples * 0.1)]
        #     elites = try_actions[inds]
        #     mean = np.mean(elites, axis=0)
        #     std = np.std(elites, axis=0)

        #print('cem act', act)

        return act

    def act(self, t=None, i_tr=None, images=None, goal_image=None, verbose_worker=None, state=None):
        self._images = images
        self._states = state[-1][:2]
        self._verbose_worker = verbose_worker
        self._t = t

        ### Support for getting goal images from environment
        if goal_image.shape[0] == 1:
          self._goal_image = goal_image[0]
        else:
          self._goal_image = goal_image[-1, 0]  # pick the last time step as the goal image
        #print(f'goal {t}: {self._goal_image}')
        import matplotlib.pyplot as plt
        #plt.imsave(f'ims/goal{t}.png', self._goal_image)
        #plt.imsave(f'ims/image{t}.png', self._images[-1][0])
        #self.get_best_action(t)
        action = {'actions': self.get_best_action(t)}
        print(action)
        return action


def ten2pytrch(img, device):
    """Converts images to the [-1...1] range of the hierarchical planner."""
    img = img[:, 0]
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)

def uint2pytorch(img, num_samples, device):
    img = np.tile(img[None], [num_samples, 1, 1, 1])
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)


