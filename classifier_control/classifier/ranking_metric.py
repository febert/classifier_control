import glob
import deepdish as dd
import glob
import numpy as np
import torch
import cv2
import tqdm


class RankingMetric:

    def __init__(self, data_dir, cache_data=True):
        self.data_dir = data_dir
        self.cache_data = cache_data
        if self.cache_data:
            self.cache = dict()

        self.img_sz = (64, 64)

    def get_traj_query(self, i, query):
        """
        :param i: index of trajectory data to get
        :return: trajectory data
        """
        t, cem_itr, horizon = query['t'], query['cem_itr'], query['horizon']
        goal_image, goal_state, idx = dd.io.load(f'{self.data_dir}/eval_data_{i}.h5', ['/goal_image', '/goal_state', '/idx'])
        assert idx == i, "File name should match index number"

        traj_images = dd.io.load(f'{self.data_dir}/eval_data_{i}.h5', f'/data/i{t}/cem_data/i{cem_itr}/traj_images', sel=dd.aslice[:, horizon-1])
        traj_states = dd.io.load(f'{self.data_dir}/eval_data_{i}.h5', f'/data/i{t}/cem_data/i{cem_itr}/traj_states', sel=dd.aslice[:, horizon-1])
        gt_scores = dd.io.load(f'{self.data_dir}/eval_data_{i}.h5', f'/data/i{t}/cem_data/i{cem_itr}/gt_scores', sel=dd.aslice[:, horizon-1])
        data = {
            'traj_images': traj_images.squeeze(),
            'traj_states': traj_states,
            'gt_scores': gt_scores,
            'goal_image': goal_image,
            'goal_state': goal_state,
        }

        return data

    @property
    def num_trajs(self):
        return len(glob.glob(f'{self.data_dir}/*'))

    @staticmethod
    def spearman_rho(scores_1, true_scores):
        from scipy.stats import spearmanr
        ranks_1, ranks_2 = scores_1.argsort().argsort(), true_scores.argsort().argsort()
        return spearmanr(ranks_1, ranks_2)[0]

    @staticmethod
    def dcg(scores_1, true_scores, normalize=True, exp=False, k=200):
        # Discounted cumulative gain
        # https://en.wikipedia.org/wiki/Discounted_cumulative_gain

        relevance = (0.3-true_scores) * 1000

        def comp_dcg(score_list):
            x = 0
            for rank, traj_idx in enumerate(score_list.argsort()[:k]):
                if exp:
                    rel_i = 2 ** (relevance[traj_idx]) - 1
                else:
                    rel_i = relevance[traj_idx]
                x += rel_i / (np.log2(rank + 2))
            return x

        if normalize:
            idgc = comp_dcg(true_scores)
        dgc = comp_dcg(scores_1)

        if normalize:
            return dgc/idgc
        else:
            return dgc

    @staticmethod
    def exp_dcg(scores_1, true_scores):
        return RankingMetric.dcg(scores_1, true_scores, exp=True)

    @staticmethod
    def kendall_tau(scores_1, true_scores, num_pairs_ret=10):
        """
        Given two np arrays of scores for the same trajectories, return the normalized Kendall tau score
        (disagreement) between them
        """
        assert len(scores_1) == len(true_scores)
        ## Double argsort gives rankings
        total = len(scores_1)
        ranks_1, ranks_2 = scores_1.argsort().argsort(), true_scores.argsort().argsort()
        disagree = 0
        agree = 0
        true_score_ordering = true_scores.argsort()
        x =0
        for number, i in enumerate(true_score_ordering): # Go through the indices based on how good they actually are
            for j in true_score_ordering[number+1:]:
                if i == j:
                    print('f')
                    continue
                if (ranks_1[i] < ranks_1[j] and ranks_2[i] > ranks_2[j]) or\
                   (ranks_1[i] > ranks_1[j] and ranks_2[i] < ranks_2[j]):
                    disagree += 1
                else:
                    agree += 1
                x += 1
        num_pairs = total * (total - 1) / 2.0
        assert x == num_pairs
        return 1.0 * (agree - disagree) / num_pairs

    @property
    def device(self):
        return torch.device('cuda')

    def query_traj_score(self, traj_data, learned_cost, metric):
        goal_image, goal_state = traj_data['goal_image'], traj_data['goal_state']

        input_images, input_states = traj_data['traj_images'], traj_data['traj_states']
        gt_scores = traj_data['gt_scores']

        batch_size = gt_scores.shape[0]

        goal_state_rep = torch.FloatTensor(goal_state).cuda()
        goal_state_rep = goal_state_rep[None].repeat(batch_size, 1)

        inp_dict = {'current_img': uint2pytorch(input_images, 0, self.device),
                    'current_state': torch.FloatTensor(input_states).cuda(),
                    'goal_state': goal_state_rep,
                    'goal_img': uint2pytorch(resample_imgs(goal_image, self.img_sz), batch_size,
                                             self.device)}
        try:
            score = learned_cost.predict(inp_dict)
        except:
            score = learned_cost(inp_dict)
        score = score.squeeze()

        if not isinstance(metric, list):
            metric = [metric]

        out = dict()
        for metric_fn in metric:
            rank_comp = metric_fn(score, gt_scores)
            out[metric_fn.__name__] = rank_comp

        return out

    def __call__(self, dist_fn, queries, scoring):
        """
        :param dist_fn: function that scores a batch of s, g pairs
        :param queries: list of dictionaries containing at least 2 keys: horizon length and 'cem_itr'
        :param scoring: scoring metric, takes two arguments (scores, true_scores)
        :return: scores in shape [num_queries, num_trajectories]
        """

        query_output_list = []

        for query in queries:
            outputs = []
            for idx in tqdm.tqdm(range(1, self.num_trajs)):
                traj_data = self.get_traj_query(idx, query)
                score = self.query_traj_score(traj_data, dist_fn, scoring)
                query_outputs = {k:np.array(v) for k, v in score.items()}
                outputs.append(query_outputs)
            outputs = self.dict_list_to_list_dict(outputs)
            query_output_list.append(outputs)
        return query_output_list

    @staticmethod
    def dict_list_to_list_dict(dict_list):
        d = dict()
        for key in dict_list[0]:
            a = []
            for traj in dict_list:
                a.append(traj[key])
            d[key] = np.stack(a)
        return d

    @staticmethod
    def get_stats(scores_dict):
        stats = dict()
        for metric_fn, scores in scores_dict.items():
            stats[metric_fn] = {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'first_quantile': np.quantile(scores, 0.25),
                'third_quantile': np.quantile(scores, 0.75),
            }
        return stats

def ten2pytrch(img, device):
    """Converts images to the [-1...1] range of the hierarchical planner."""
    img = img[:, 0]
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)

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

def uint2pytorch(img, num_samples, device):
    if num_samples != 0:
        img = np.tile(img[None], [num_samples, 1, 1, 1])
    img = np.transpose(img, [0, 3, 1, 2])
    return torch.from_numpy(img * 2 - 1.0).float().to(device)

class RandomScorer:
    def predict(self, inp_dict):
        return np.random.random(inp_dict['current_state'].shape[0])

if __name__ == '__main__':
    import os
    from classifier_control.classifier.utils.DistFuncEvaluation import DistFuncEvaluation
    from classifier_control.classifier.models.q_function import QFunctionTestTime
    from classifier_control.classifier.models.dist_q_function import DistQFunctionTestTime
    from classifier_control.classifier.models.tempdist_regressor import TempdistRegressor
    from classifier_control.baseline_costs.image_mse_cost import ImageMseCost

    data_dir = f'{os.environ["VMPC_EXP"]}/planning_eval/'

    model_path = \
        os.environ[
            'VMPC_EXP'] + '/classifier_control/distfunc_training/q_function_training/tabletop-texture/goal_cql/weight=100_batchfix_ONLYCQL/weights/weights_ep100.pth'
    #
    # model_path = \
    #      os.environ[
    #          'VMPC_EXP'] + '/classifier_control/distfunc_training/q_function_training/tabletop-texture/' + \
    #          'only_arm_state_loopback/weights/weights_ep995.pth'
    #
    # model_path = \
    #      os.environ[
    #          'VMPC_EXP'] + '/classifier_control/distfunc_training/q_function_training/tabletop-texture/' + \
    #          'lowdim_loopback_endgoal_negs/weights/weights_ep995.pth'

    # model_path = \
    #     os.environ[
    #         'VMPC_EXP'] + '/classifier_control/distfunc_training/q_function_training/tabletop-texture/' + \
    #     'lowdim/weights/weights_ep995.pth'

    model_path = \
         os.environ[
             'VMPC_EXP'] + '/classifier_control/distfunc_training/q_function_training/tabletop-texture/' + \
         'inventory/l2_obj_rew_ac_lowdim/weights/weights_ep600.pth'

    learned_cost_class = QFunctionTestTime
    #learned_cost_class = ImageMseCost

    rm = RankingMetric(data_dir, cache_data=False)
    learned_cost_testparams = dict()
    learned_cost_testparams['data_conf'] = {'img_sz': (64, 64)}  # todo currently uses 64x64!!
    learned_cost_testparams['classifier_restore_path'] = model_path
    learned_cost = DistFuncEvaluation(learned_cost_class, learned_cost_testparams)

    queries = [
        {
            't': 0,
            'cem_itr': 0,
            'horizon': 13,
        }
    ]

    import time
    start_time = time.time()
    scores = rm(RandomScorer(), queries=queries, scoring=[rm.kendall_tau, rm.dcg, rm.exp_dcg])
    print("--- %s seconds ---" % (time.time() - start_time))
    print((scores[0]))
    scores = rm(learned_cost, queries=queries, scoring=[rm.kendall_tau, rm.dcg, rm.exp_dcg])
    print("--- %s seconds ---" % (time.time() - start_time))

    print(rm.get_stats(scores[0])['exp_dcg'])



