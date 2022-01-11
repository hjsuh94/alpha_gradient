import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import torch

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

class AngularTorch(ObjectiveFunction):
    def __init__(self):
        super().__init__(1)
        self.d = 1
        self.theta_max = np.deg2rad(45)

    def evaluate(self, x, w):
        z = x + w
        if (np.abs(z) <= self.theta_max):
            return -np.sin(z) ** 2.0
        else:
            return 0.0

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        B = w.shape[0]

        x = torch.Tensor(x)
        w = torch.Tensor(w)

        z = x + w
        cost_array = torch.zeros(B,1)

        hit_ind = (torch.abs(z) <= self.theta_max).bool()
        miss_ind = (torch.abs(z) > self.theta_max).bool()

        cost_array[hit_ind] = -torch.pow(torch.sin(z), 2)[hit_ind]
        cost_array[miss_ind] = 0.0

        return cost_array.detach().cpu().numpy()
