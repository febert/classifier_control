import faiss
import numpy as np
import torch

class NNIndex:

    nlist = 100
    dim = 18
    num_samples = 200000

    def __init__(self, dataloader, gpu_id, bs, low_dim):
        self.low_dim = low_dim
        self.gpu_res = faiss.StandardGpuResources()
        self.index = faiss.IndexIDMap2(faiss.IndexIVFFlat(faiss.IndexFlatL2(self.dim), self.dim, self.nlist))
        if gpu_id == -1:
            gpu_id = 0
        self.index = faiss.index_cpu_to_gpu(self.gpu_res, gpu_id, self.index)

        self.dataset = dataloader.dataset
        self.T = self.dataset.T
        self.dataloader = dataloader
        self.cache = {}
        self.build_index()

    def lookup(self, idx):
        res = []
        eeps = []
        for lookup in idx:
            #pt = self.dataset[lookup//self.T]
            pt = self.cache[lookup]
            res.append(pt)
            #res.append(pt['states'][lookup%self.T])
            #eeps.append(pt['gripper'][lookup%self.T])
        return torch.FloatTensor(np.stack(res)).cuda()

    def find_knn(self, queries, k=2):
        # Queries should be shape (batch, self.d)
        # K: number of nn to return
        D, I = self.index.search(queries.cpu().numpy(), k)
        return I

    def build_index(self):
        samples = []
        indices = []
        # Index code will be traj_idx * T + idx
        num_trajs = len(self.dataloader)
        samp_per = self.num_samples // (32 * num_trajs) + 1
        for batch in self.dataloader:
            bs = batch['states'].shape[0]

            rand = torch.rand(self.T, bs)
            sampled_times = rand.argsort(dim=0)[:samp_per]

            for sample in sampled_times:
                state_size = batch['states'].shape[-1]
                arm_states = torch.cat((batch['states'][..., :9], batch['states'][..., state_size//2:state_size//2+9]), dim=-1)
                #arm_states = batch['states'][..., :9]
                samples.append(arm_states[np.arange(bs), sample])
                ind = batch['index'] * self.T + sample
                #samples.append(arm_states[torch.arange(bs), sampled_times])
                indices.append(ind)
                for i in range(bs):
                    if self.low_dim:
                        self.cache[ind[i].item()] = batch['states'][i, sample[i]]
                    else:
                        self.cache[ind[i].item()] = batch['images'][i, sample[i]]

        self.samples, self.indices = torch.cat(samples), torch.cat(indices)
        print(f'Training FAISS on {self.samples.shape[0]} points...')
        self.index.train(self.samples.numpy())
        print('Adding points...')
        self.index.add_with_ids(self.samples.numpy(), self.indices.numpy())
        print('Done...')












