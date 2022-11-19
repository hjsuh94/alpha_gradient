import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import torch

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm

def ball_trajectory(v0, theta, d, h, g, get_trajectory=False):
    xt = 2 * (v0 ** 2) * np.sin(theta) * np.cos(theta) / g
    if xt <= d:
        x_final = xt 
    else:
        ball_height = -0.5 * g / (v0 ** 2.0) * (1 / (
            np.cos(theta) ** 2.0)) * (d ** 2.0) + np.tan(theta) * d
        if ball_height <= h:
            x_final = d * (xt / xt)
        else:
            x_final = xt

    if get_trajectory:
        x_trj = np.linspace(0, x_final, 1000)
        y_trj = -0.5 * g / (v0 ** 2.0) * (1 / (
            np.cos(theta) ** 2.0)) * (x_trj ** 2.0) + np.tan(theta) * x_trj
        trj = np.vstack((x_trj, y_trj))
    else:
        trj = None

    return x_final, trj

class BallWithWallSoftPlusObjective(ObjectiveFunction):
    def __init__(self, dynamics):
        super().__init__(1)
        self.dynamics = dynamics
        self.d = 1

    def evaluate(self, x, w):
        trj = self.dynamics.rollout(x + w)
        return -trj[-1,0] ** 2.0

    def evaluate_batch(self, x, w):
        B = w.shape[0]
        trjs = self.dynamics.rollout_batch(x + w)
        return -trjs[:,-1,0] ** 2.0

    def gradient(self, x, w):

        z = torch.tensor(x + w, requires_grad=True, dtype=torch.float32)
        trj = self.dynamics.rollout(z)
        cost = trj[-1, 0]
        cost.backward()
        return z.grad.detach().numpy()

    def gradient_batch(self, x, w):
        B = w.shape[0]        
        z = torch.tensor(x + w, requires_grad=True, dtype=torch.float32)
        trjs = self.dynamics.rollout_batch(z.squeeze(1))
        cost = torch.sum(-trjs[:,-1,0] ** 2.0)
        cost.backward()
        return z.grad.detach().numpy()

    def zero_order_gradient_batch(self, x, w, stdev):
        """
        Evaluate zero-order gradient batch.
        input: x of shape n, w of shape (B, m).
        output: dfdx^0(x) of shape (B, n).
        """
        B = w.shape[0]
        # This should be of shape B.
        cost = self.evaluate_batch(x, w.squeeze()) - self.evaluate_batch(
            x, np.zeros(B))
        # This should be of shape B x m
        return np.multiply(cost[:,None], w) / (stdev ** 2.0)

    def zobg_given_samples(self, x, samples, stdev):
        """
        Compute zero order batch gradient GIVEN the samples directly.
        Sample must be of shape (N, D) where N is sample size, D is 
        dimension of underlying data. 
        """
        batch = self.zero_order_gradient_batch(x, samples, stdev).detach().numpy()
        return compute_mean(batch), compute_variance_norm(batch, 2)