import numpy as np
import torch 

from alpha_gradient.dynamical_system import DynamicalSystem

# Simple discrete-time linear dynamical system.

class LinearDynamics(DynamicalSystem):
    def __init__(self, d):
        super().__init__()
        self.h = 0.05
        self.dim_x = d
        self.dim_u = d

        self.A = torch.eye(self.dim_x)
        self.B = torch.eye(self.dim_u)

    def dynamics(self, x, u):
        return self.A @ x + self.B @ u

    def dynamics_batch(self, x_batch, u_batch):
        # x : B x 2
        # u : B x 2
        
        return x_batch @ self.A + u_batch @ self.B

    def rollout(self, x0, u_trj):
        """
        Rollout system from x0 to u_trj using dynamics.
        args:
        - x0 (torch.Tensor, dim: n): initial states.
        - u_trj (torch.Tensor, dim: T x m): input trajectories.
        returns:
        - x_trj (torch.Tensor, dim: (T+1) x n): batch of state trajs.
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
        - x0 (torch.Tensor, dim: B x n): batch of initial states.
        - u_trj (torch.Tensor, dim: B x T x m): batch of input trajectories.
        returns:
        - x_trj (torch.Tensor, dim: B x (T+1) x n): batch of state trajs.
        """
        B = u_trj.shape[0]
        T = u_trj.shape[1]
        x_trj = torch.zeros((B, T+1, self.dim_x))
        x_trj[:,0,:] = x0
        for t in range(T):
            x_trj[:,t+1,:] = self.dynamics_batch(x_trj[:,t,:], u_trj[:,t,:])
        return x_trj        

def test_linear_dynamics():
    d = 5
    dynamics = LinearDynamics(d)
    dynamics.dynamics(torch.ones(d), torch.ones(d))
    dynamics.dynamics_batch(torch.ones(10,d), torch.ones(10,d))
    dynamics.rollout(torch.ones(d), torch.ones(100,d))
    dynamics.rollout_batch(torch.ones(7,d), torch.ones(7, 10,d))
