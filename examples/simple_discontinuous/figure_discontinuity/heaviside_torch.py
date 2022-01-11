import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import torch

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

class HeavisideTorch(ObjectiveFunction):
    def __init__(self):
        super().__init__(1)
        self.d = 1

    def evaluate(self, x, w):
        if (x + w) >= 0:
            return 1 
        else:
            return 0

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)
        B = w.shape[0]

        x = torch.Tensor(x)
        w = torch.Tensor(w)

        z = x + w
        cost_array = torch.zeros(B,1)

        cost_array[z >= 0] = 1.0
        cost_array[z < 0] = 0.0

        return cost_array.detach().cpu().numpy()

