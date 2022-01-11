import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import torch

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

def ball_trajectory(v0, theta, d, h, g, get_trajectory=False):
    xt = 2 * (v0 ** 2) * torch.sin(theta) * torch.cos(theta) / g
    ball_height = -0.5 * g / (v0 ** 2.0) * (1 / (
        torch.cos(theta) ** 2.0)) * (d ** 2.0) + torch.tan(theta) * d

    B = theta.shape[0]
    x_final = torch.zeros(B,1)

    x_final[xt <= d] = xt[xt <= d]
    x_final[np.logical_and((xt > d), (ball_height <= h))] = d
    x_final[np.logical_and((xt > d), (ball_height > h))] = xt[
        np.logical_and((xt > d), (ball_height > h))]

    return x_final

class BallWithWallTorch(ObjectiveFunction):
    def __init__(self):
        super().__init__(1)
        self.d = 1
        self.v0 = 1
        self.dball = 0.06
        self.hball = 0.02
        self.gball = 9.81

    def evaluate(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        x = torch.Tensor(x)
        w = torch.Tensor(w).unsqueeze(0)

        x_final = ball_trajectory(
            self.v0, x+w, self.dball, self.hball, self.gball)
        cost_array = -torch.pow(x_final, 2.0).squeeze(0)
        return cost_array.detach().cpu().numpy()

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)
        B = w.shape[0]

        x = torch.Tensor(x)
        w = torch.Tensor(w)

        x_final = ball_trajectory(
            self.v0, x+w, self.dball, self.hball, self.gball)
        cost_array = -torch.pow(x_final, 2.0).squeeze(1)
        return cost_array.detach().cpu().numpy()

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        x = torch.Tensor(x, requires_grad=True)
        w = torch.Tensor(w)

        x_autodiff = InitializeAutoDiff(x+w)
        x_final, trj = ball_trajectory(
            self.v0, x_autodiff, self.dball, self.hball, self.gball)
        x_final_cost = -x_final ** 2.0
        dfdx = ExtractGradient(x_final_cost)
        return dfdx

    def gradient_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        B = w.shape[0]

        z = torch.tensor(x + w, requires_grad=True, dtype=torch.float32)
        x_final = ball_trajectory(
            self.v0, z, self.dball, self.hball, self.gball)
        cost_array = -torch.sum(torch.pow(x_final, 2.0))

        cost_array.backward()
        return z.grad.detach().cpu().numpy()

"""
obj = BallWithWallTorch()
print(obj.evaluate(np.array([0.5]), np.array([0.0])))
print(obj.evaluate_batch(np.array([0.5]), np.random.normal(0, 0.1, (10,1))))
print(obj.gradient_batch(np.array([0.5]), np.random.normal(0, 0.1, (10,1))))
"""