import numpy as np
import torch

from dynamical_system import DynamicalSystem

class DynamicalSystemTorch(DynamicalSystem):
    def __init__(self):
        super().__init__()

    def rollout(self, x0, u_trj):
        """
        Rollout system from x0 to u_trj using dynamics.
        args:
        - x0 (torch.Tensor, dim: n): initial state
        - u_trj (torch.Tensor, dim: T x m): input trajectory.
        returns:
        - x_trj (torch.Tensor, dim: (T+1) x n): resulting state trajectory.
        """
        T = u_trj.shape[0]
        x_trj = torch.zeros((T+1, self.dim_x))
        x_trj[0,:] = x0
        for t in range(T):
            x_trj[t+1,:] = self.dynamics(x_trj[t,:], u_trj[t,:])
        return x_trj

    def rollout_batch(self, x0, u_trj):
        """
        Rollout system from x0 to u_trj using dynamics in batch.
        args:
        - x0 (np.array, dim: B x n): batch of initial states.
        - u_trj (torch.Tensor, dim: B x T x m): batch of input trajectories.
        returns:
        - x_trj (torch.Tensor, dim: B x (T+1) x n): batch of states trajs.
        """
        B = u_trj.shape[0]
        T = u_trj.shape[1]
        assert (x0.shape[0] == u_trj.shape[0])
        x_trj = torch.zeros((B, T+1, self.dim_x))
        x_trj[:,0,:] = x0
        for t in range(T):
            x_trj[:,t+1,:] = self.dynamics_batch(x_trj[:,t,:], u_trj[:,t,:])
        return x_trj
