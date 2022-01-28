import numpy as np
import matplotlib.pyplot as plt

from alpha_gradient.trajectory_optimizer import (
    TrajoptParameters, TrajectoryOptimizer)
from alpha_gradient.numpy.trajectory_optimizer_np import (
    TrajectoryOptimizerNp
)

class CemNpParams(TrajoptParameters):
    def __init__(self):
        super().__init__()
        self.n_elite = None
        self.batch_size = None 
        self.initial_std = None # dim u array of initial stds.

class CemNp(TrajectoryOptimizerNp):
    def __init__(self, system, params):
        super().__init__(system, params)
        
        self.n_elite = self.params.n_elite
        self.batch_size = self.params.batch_size
        self.std_trj = self.params.initial_std

    def local_descent(self, x_trj, u_trj):
        """
        Given batch of x_trj and u_trj, run forward pass of algorithm.
        - args:
            x_trj (np.array, shape (T + 1) x n): nominal state trajectory.
            u_trj (np.array, shape T x m) : nominal input trajectory
        """

        # 1. Produce candidate trajectories according to u_std.
        u_trj_mean = u_trj
        u_trj_candidates = np.random.normal(u_trj_mean, self.std_trj,
            (self.params.batch_size, self.T, self.dim_u))
        cost_array = np.zeros(self.batch_size)

        # 2. Roll out the trajectories.
        for k in range(self.batch_size):
            u_trj_cand = u_trj_candidates[k,:,:]
            cost_array[k] = self.evaluate_cost(
                self.system.rollout(self.x0, u_trj_cand), u_trj_cand)

        # 3. Pick the best K trajectories.
        # In the reward setting, this is the highest. In cost, it's lowest.
        best_idx = np.argpartition(cost_array, self.n_elite)[:self.n_elite]

        best_trjs = u_trj_candidates[best_idx,:,:]

        # 4. Set mean as the new trajectory, and update std.
        u_trj_new = np.mean(best_trjs, axis=0)
        u_trj_std_new = np.std(best_trjs, axis=0)
        self.std_trj = u_trj_std_new
        x_trj_new = self.system.rollout(self.x0, u_trj_new)

        return x_trj_new, u_trj_new