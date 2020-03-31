import numpy as np
import torch


class MixupRegularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x1, x2):
        lam = np.random.beta(self.alpha, self.alpha, x1.shape[0])
        lam = torch.from_numpy(lam).float().to('cuda')
        return self.convex_comb(x1, x2, lam), lam

    @staticmethod
    def convex_comb(x1, x2, lam):
        assert x1.shape == x2.shape, 'convex comb requires same shape!'
        lam_shape = [1] * len(x1.shape)
        lam_shape[0] = -1
        lam = lam.view(*lam_shape)
        return lam * x1 + (1 - lam) * x2
