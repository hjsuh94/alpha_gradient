import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import autograd

from alpha_gradient.trajectory_optimizer import (
    TrajoptParameters, TrajectoryOptimizer)
from alpha_gradient.torch.trajectory_optimizer_torch import (
    TrajectoryOptimizerTorch
)    

class ZobgdTorchParams(TrajoptParameters):
    def __init__(self):
        """
        Variance scheduler is a function with
        args:
        -iter: current iteration
        -var : initial variance
        returns:
        -var : current variance.
        """
        super().__init__()
        self.step_size = None # Step size for gradietn descent.
        self.batch_size = None # Number of samples used for estimation.
        self.initial_std = None # dim T x m array of initial stds.
        self.variance_scheduler = None # Variance scheduler.
        self.stepsize_scheduler = None
        self.gpu = False

class ZobgdTorch(TrajectoryOptimizerTorch):
    def __init__(self, system, params):
        super().__init__(system, params)
        
        self.initial_stepsize = self.params.step_size
        self.batch_size = self.params.batch_size
        self.initial_std = self.params.initial_std
        self.variance_scheduler = self.params.variance_scheduler
        self.stepsize_scheduler = self.params.stepsize_scheduler

        self.step_size = self.initial_stepsize
        self.w_std = self.initial_std

    def compute_zobg(self, x_trj, u_trj):
        # 1. Take samples to perturb decision variables (u_trj).
        B = self.batch_size

        w_batch = torch.Tensor(np.random.normal(
            0, self.w_std, (B, self.T, self.dim_u)))
        u_trj_batch = u_trj + w_batch
        x_trj_batch = self.system.rollout_batch(self.x0, u_trj_batch)

        cost_batch = self.evaluate_cost_batch(x_trj_batch, u_trj_batch)
        cost_zero = self.evaluate_cost(x_trj, u_trj)
        cost_diff = cost_batch - cost_zero 

        zobg = torch.mean(
            cost_diff[:,None,None] * w_batch, axis=0) / torch.pow(self.w_std, 2.)

        return zobg


    def local_descent(self, x_trj, u_trj):
        """
        Given batch of x_trj and u_trj, run forward pass of algorithm.
        - args:
            x_trj (np.array, shape (T + 1) x n): nominal state trajectory.
            u_trj (np.array, shape T x m) : nominal input trajectory
        """
        zobg = self.compute_zobg(x_trj, u_trj)

        u_trj_new = u_trj - self.step_size * zobg
        x_trj_new = self.system.rollout(self.x0, u_trj_new)
        self.w_std = self.variance_scheduler(self.iter, self.initial_std)
        self.step_size = self.stepsize_scheduler(self.iter, self.initial_stepsize)

        return x_trj_new, u_trj_new
