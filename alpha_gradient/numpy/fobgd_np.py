import numpy as np
import matplotlib.pyplot as plt

from alpha_gradient.trajectory_optimizer import (
    TrajoptParameters, TrajectoryOptimizer)
from alpha_gradient.numpy.trajectory_optimizer_np import (
    TrajectoryOptimizerNp
)    
from pydrake.all import InitializeAutoDiff, ExtractGradient
import pydrake.autodiffutils

class FobgdNpParams(TrajoptParameters):
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
        self.batch_size = None # Number of samples used for estimation.
        self.initial_std = None # dim T x m array of initial stds.
        self.variance_scheduler = None # Variance scheduler.
        self.stepsize_scheduler = None

class FobgdNp(TrajectoryOptimizerNp):
    def __init__(self, system, params):
        super().__init__(system, params)
        
        self.batch_size = self.params.batch_size
        self.initial_std = self.params.initial_std
        self.variance_scheduler = self.params.variance_scheduler
        self.stepsize_scheduler = self.params.stepsize_scheduler
        self.w_std = self.initial_std

    def compute_fobg(self, u_trj):
        """
        NOTE (terry-suh): Eigen autodiff hates batches, so we're unfortunately 
        limited to for loops. Batches are technically doable with a lot of waste
        in memory, should investigate later.
        """
        # 1. Take samples to perturb decision variables (u_trj).
        B = self.batch_size
        w_batch = np.random.normal(0, self.w_std, (B, self.T, self.dim_u))
        u_trj_batch = u_trj + w_batch

        # 2. Compute gradients.
        fobg = np.zeros((self.T, self.dim_u))
        for b in range(B):
            u_trj = InitializeAutoDiff(u_trj_batch[b,:,:])
            x_trj = self.system.rollout(self.x0, u_trj,
                dtype=pydrake.autodiffutils.AutoDiffXd)
            cost = self.evaluate_cost(x_trj, u_trj)
            dfdu = ExtractGradient(np.array([cost]))
            fobg += dfdu.transpose()
        fobg /= B
        return fobg

    def local_descent(self, x_trj, u_trj):
        """
        Given batch of x_trj and u_trj, run forward pass of algorithm.
        - args:
            x_trj (np.array, shape (T + 1) x n): nominal state trajectory.
            u_trj (np.array, shape T x m) : nominal input trajectory
        """
        # 1. Compute gradient.
        fobg = self.compute_fobg(u_trj)

        # 2. Determine adequate stepsize.
        step_size = self.stepsize_scheduler.find_stepsize(
            self.objective, fobg, u_trj)
        u_trj_new = u_trj - step_size * fobg
        x_trj_new = self.system.rollout(self.x0, u_trj_new)
        self.w_std = self.variance_scheduler(self.iter, self.initial_std)
        self.stepsize_scheduler.step()

        return x_trj_new, u_trj_new
