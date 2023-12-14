import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from tqdm import tqdm
import torch

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

class PivotObjective(ObjectiveFunction):
    def __init__(self, dynamics):
        super().__init__(1)
        self.dynamics = dynamics
        self.T = 200
    
    def evaluate_cost(self, x_trj):
        """
        Evaluate cost given an state-input trajectory.
        - args:
            x_trj (shape (T + 1) x n): state trajectory
            u_trj (shape T x m): input trajectory
        """
        return -x_trj[-1,5] ** 2.0

    def evaluate_cost_batch(self, x_trj):
        """
        Evaluate cost given a batch of state-input trajectories.
        -args:
        x_trj (shape: B x (T+1) x n): state trajectory.
        u_trj (shape: B x T x m): input trajectory.
        """
        return -x_trj[:,-1,5] ** 2.0

    def evaluate(self, x, w):
        """
        x: input array of shape (T x m,) (has to be one-dimensional)
        """
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        z = torch.tensor(x + w, dtype=torch.float32)
        x0 = torch.tensor([0.0, z])
        x_trj = self.dynamics.rollout(x0, self.T)
        cost = self.evaluate_cost(x_trj)
        return cost.detach().numpy()


    def evaluate_batch(self, x, w):
        """
        x: input array of shape (T x m,) (has to be one-dimensional)
        w: array of shape (B, T x m)
        """
        #assert(len(x) == self.d)
        #assert(w.shape[1] == self.d)

        B = w.shape[0]
        vx = self.dynamics.v0 * torch.cos(x)
        vy = self.dynamics.v0 * torch.sin(x)
        x0 = torch.hstack([torch.zeros((B,3)), vx[:,None], vy[:,None], torch.zeros((B,1))])

        x_trj = self.dynamics.rollout_batch(x0)
        cost = self.evaluate_cost_batch(x_trj)
        return cost.detach().numpy()

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        z = torch.tensor(x + w, requires_grad=True, dtype=torch.float32)
        u_trj = torch.zeros(1)
        x0 = torch.tensor([0.0, z])

        x_trj = self.dynamics.rollout(x0, u_trj)
        cost = self.evaluate_cost(x_trj, u_trj)
        cost.backward()
        return z.grad.detach().numpy()

    def gradient_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)        

        B = w.shape[0]
        z = torch.tensor(x + w, requires_grad=True, dtype=torch.float32)
        u_trj = torch.zeros(1)
        x0 = torch.tensor([0.0, z])

        x_trj = self.dynamics.rollout_batch(x0, u_trj)
        cost = torch.sum(self.evaluate_cost_batch(x_trj, u_trj))
        cost.backward()
        return z.grad.detach().numpy()
