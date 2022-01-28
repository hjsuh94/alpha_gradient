import numpy as np
import matplotlib.pyplot as plt

import torch

from alpha_gradient.trajectory_optimizer import (
    TrajoptParameters, TrajectoryOptimizer)
from alpha_gradient.torch.trajectory_optimizer_torch import (
    TrajectoryOptimizerTorch
)

class CemTorchParams(TrajoptParameters):
    def __init__(self):
        super().__init__()
        self.n_elite = None
        self.batch_size = None 
        self.initial_std = None # dim u array of initial stds.
        self.gpu = False

class CemTorch(TrajectoryOptimizerTorch):
    def __init__(self, system, params):
        super().__init__(system, params)
        
        self.n_elite = self.params.n_elite
        self.batch_size = self.params.batch_size
        self.std_trj = self.params.initial_std
        self.gpu = self.params.gpu

    def local_descent(self, x_trj, u_trj, gpu=False):
        """
        Given batch of x_trj and u_trj, run forward pass of algorithm.
        - args:
            x_trj (np.array, shape (T + 1) x n): nominal state trajectory.
            u_trj (np.array, shape T x m) : nominal input trajectory
        """

        # 1. Produce candidate trajectories according to u_std.
        u_trj_mean = u_trj

        if(self.gpu):
            u_trj_candidates = torch.Tensor(
                self.params.batch_size, self.T, self.dim_u)
            for b in range(self.params.batch_size):
                u_trj_candidates[b,:,:] = torch.normal(
                    u_trj_mean, self.std_trj)
        else:
            u_trj_candidates = torch.Tensor(np.random.normal(
                u_trj_mean.numpy(), self.std_trj.numpy(),
                (self.params.batch_size, self.T, self.dim_u)))
        cost_array = torch.zeros(self.batch_size)

        if (self.params.gpu):
            u_trj_candidates = u_trj_candidates.cuda()
            cost_array = cost_array.cuda()

        # 2. Roll out the trajectories.
        cost_array = self.evaluate_cost_batch(
            self.system.rollout_batch(
                self.x0, u_trj_candidates, self.gpu),
                u_trj_candidates, self.gpu)

        # 3. Pick the best K trajectories.
        best_idx = torch.topk(cost_array, self.n_elite, largest=False)[1]
        best_trjs = u_trj_candidates[best_idx,:,:]

        # 4. Set mean as the new trajectory, and update std.
        u_trj_new = torch.mean(best_trjs, axis=0)
        u_trj_std_new = torch.std(best_trjs, axis=0)
        self.std_trj = u_trj_std_new
        x_trj_new = self.system.rollout(self.x0, u_trj_new, self.gpu)

        return x_trj_new, u_trj_new
