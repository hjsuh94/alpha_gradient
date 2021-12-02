import numpy as np
import matplotlib.pyplot as plt

from alpha_gradient.trajectory_optimizer import (
    TrajoptParameters, TrajectoryOptimizer)

class TrajectoryOptimizerNp(TrajectoryOptimizer):
    def __init__(self, system, params):
        super().__init__(system, params)

    def evaluate_cost(self, x_trj, u_trj):
        """
        Evaluate cost given an state-input trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n): state trajectory
            u_trj (np.array, shape T x m): input trajectory
        """
        cost = 0.0
        for t in range(self.T):
            et = x_trj[t, :] - self.xd_trj[t, :]
            cost += et.dot(self.Q).dot(et)
            cost += (u_trj[t, :]).dot(self.R).dot(u_trj[t, :])
        et = x_trj[self.T, :] - self.xd_trj[self.T, :]
        cost += et.dot(self.Qd).dot(et)
        return cost

    def evaluate_cost_batch(self, x_trj, u_trj):
        """
        Evaluate cost given a batch of state-input trajectories.
        -args:
        x_trj (np.array, shape: B x (T+1) x n): state trajectory.
        u_trj (np.array, shape: B x T x m): input trajectory.
        """
        B = x_trj.shape[0]
        cost = np.zeros(B)
        for t in range(self.T):
            et = x_trj[:, t, :] - self.xd_trj[t, :]
            cost += np.diagonal(et.dot(self.Q).dot(et.transpose()))
            ut = u_trj[:,t,:]
            cost += np.diagonal(ut.dot(self.R).dot(ut.transpose()))            
        et = x_trj[:, self.T, :] - self.xd_trj[self.T, :]
        cost += np.diagonal(et.dot(self.Qd).dot(et.transpose()))        
        return cost        
