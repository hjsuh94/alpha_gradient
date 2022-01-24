import os, shutil, subprocess

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from alpha_gradient.dynamical_system import DynamicalSystem

class FrictionDynamics(DynamicalSystem):
    def __init__(self, vs, mu):
        super().__init__()

        self.h = 0.01
        self.dim_x = 4
        self.dim_u = 1

        # Geometry parameters.
        self.mass = 0.1
        self.width = 0.1

        self.vs = vs
        self.mu = mu

    def dynamics(self, x, u):
        x1_now = x[0].clone()
        x2_now = x[1].clone()
        v1_now = x[2].clone()
        v2_now = x[3].clone()
        u_now = u[0].clone()

        # Compute forces.
        x_diff = x2_now - x1_now
        v_diff = v2_now - v1_now

        in_contact = torch.tensor((torch.abs(x_diff) < self.width)).float()

        friction = -in_contact * torch.clamp(self.mu * v_diff,
            min= -self.mu * self.vs, max = self.mu * self.vs)

        f1_now = -friction + u_now
        f2_now = friction

        # Damping on velocity.
        v1_next = v1_now + self.h / self.mass * f1_now
        v2_next = v2_now + self.h / self.mass * f2_now
        x1_next = x1_now + self.h * v1_next
        x2_next = x2_now + self.h * v2_next


        return torch.hstack((x1_next, x2_next, v1_next, v2_next))

    def dynamics_batch(self, x_batch, u_batch):
        """
        x_batch: torch.Tensor, shape: (B x n)
        u_batch: torch.Tensor, shape: (B x 1)
        """
        x1_now = x_batch[:,0].clone()
        x2_now = x_batch[:,1].clone()
        v1_now = x_batch[:,2].clone()
        v2_now = x_batch[:,3].clone()
        u_now = u_batch[:,0].clone()

        # Compute forces.
        x_diff = x2_now - x1_now
        v_diff = v2_now - v1_now

        in_contact = torch.tensor((torch.abs(x_diff) < self.width)).float()

        friction = -in_contact * torch.clamp(self.mu * v_diff,
            min= -self.mu * self.vs, max = self.mu * self.vs)

        f1_now = -friction + u_now
        f2_now = friction

        # Damping on velocity.
        v1_next = v1_now + self.h / self.mass * f1_now
        v2_next = v2_now + self.h / self.mass * f2_now
        x1_next = x1_now + self.h * v1_next
        x2_next = x2_now + self.h * v2_next

        return torch.vstack((x1_next, x2_next, v1_next, v2_next)).transpose(0,1)

    def rollout(self, x0, u_trj):
        """
        Rollout system from x0 to u_trj using dynamics.
        args:
        - x0 (torch.Tensor, dim: n): initial states.
        - u_trj (torch.Tensor, dim: T x m): input trajectories.
        returns:
        - x_trj (torch.Tensor, dim: (T+1) x n): batch of state trajs.
        """
        u_trj = torch.tensor(u_trj)
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

def test_friction_dynamics():
    dynamics = FrictionDynamics(0.01, 0.01)

    T = 4000

    x0 = torch.tensor([0.0, 0.0, 0.001, 0.0])
    u_trj = 0.00001 * torch.ones(T, 1)
    u_trj[2000:4000] = 0.0
    x_trj = dynamics.rollout(x0, u_trj)

    x0_batch = torch.zeros(1000, 4)
    x0_batch[:,1] = 100.0
    u_trj_batch = 30.0 * torch.rand(1000, T, 1)
    x_trj_batch = dynamics.rollout_batch(x0_batch, u_trj_batch)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(x_trj[:,0])
    plt.plot(x_trj[:,1])
    plt.subplot(1,2,2)
    plt.plot(x_trj[:,2])
    plt.plot(x_trj[:,3])    
    plt.show()
