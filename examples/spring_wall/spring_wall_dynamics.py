import os, shutil, subprocess

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from alpha_gradient.dynamical_system import DynamicalSystem

class SpringWallDynamics(DynamicalSystem):
    def __init__(self, k, d):
        super().__init__()

        self.h = 0.001
        self.dim_x = 2
        self.dim_u = 1

        # Geometry parameters.
        self.mass = 0.1
        self.stiffness = k
        self.damping = d

    def dynamics(self, x, u):
        x_now = x[0].clone()
        v_now = x[1].clone()

        # Semi-implicit integration.

        if x_now > 1.0:
            f = -self.stiffness * (x_now - 1.0) - self.damping * v_now
        elif x_now < -1.0:
            f = -self.stiffness * (x_now + 1.0) - self.damping * v_now
        else:
            f = u

        v_next = v_now + self.h / self.mass * f
        x_next = x_now + self.h * v_next

        return torch.hstack((x_next, v_next))

    def dynamics_batch(self, x_batch, u_batch):
        """
        x_batch: torch.Tensor, shape: (B x n)
        u_batch: torch.Tensor, shape: (B x 1)
        """

        x_now = x_batch[:,0].clone()
        v_now = x_batch[:,1].clone()
        u_now = u_batch[:,0].clone()

        col_ind = (torch.abs(x_now) > 1.0).float()

        f = col_ind * (
            -self.stiffness * (
                x_now - torch.sign(
                    x_now).float()) - self.damping * v_now) + u_now

        v_next = v_now + self.h / self.mass * f
        x_next = x_now + self.h * v_next

        return torch.vstack((x_next, v_next)).transpose(0,1)

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

def test_springwall_dynamics():
    dynamics = SpringWallDynamics(10000.0, 3.0)

    T = 2000

    x0 = torch.tensor([0.0, 100.0])
    u_trj = 0.0 * torch.ones(T, 1)

    x_trj = dynamics.rollout(x0, u_trj)

    x0_batch = torch.zeros(1000, 2)
    x0_batch[:,1] = 100.0
    u_trj_batch = 30.0 *torch.rand(1000, T, 1)
    x_trj_batch = dynamics.rollout_batch(x0_batch, u_trj_batch)
