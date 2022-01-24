import os, shutil, subprocess

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from alpha_gradient.dynamical_system import DynamicalSystem

class DoublePendulumDynamics(DynamicalSystem):
    def __init__(self):
        super().__init__()

        self.h = 0.001
        self.dim_x = 4
        self.dim_u = 1

        # Geometry parameters.
        self.m = 0.1
        self.l = 1.0
        self.inertia = self.m * (self.l ** 2.0)
        self.g = 9.81

    def dynamics(self, x, u):
        q1_now = x[0].clone()
        q2_now = x[1].clone()
        q1dot_now = x[2].clone()
        q2dot_now = x[3].clone()

        # Semi-implicit integration.

        # 1. Compute q2ddot

        q1ddot = (
            -self.g * (3 * self.m) * torch.sin(q1_now) - self.m * self.g * (
                torch.sin(q1_now - 2 * q2_now)) - 2 * torch.sin(
                    q1_now - q2_now) * (self.m * (
                        torch.pow(
                            q2dot_now, 2) * self.l + torch.pow(
                                q1dot_now, 2) * self.l * torch.cos(
                                    q1_now - q2_now)
                ))) / (
                    self.l * self.m * (3 - torch.cos(2 * (q1_now - q2_now))))

        q2ddot = (
            2 * torch.sin(q1_now - q2_now) * (torch.pow(
                q1_now, 2) * self.l * self.m * 2.0 + 2.0 * self.m * self.g * (
                    torch.cos(q1_now) + torch.pow(
                        q2dot_now, 2) * self.l * self.m * torch.cos(
                            q1_now - q2_now)
                )) / (
                    self.l * self.m * (3 - torch.cos(2 * (q1_now - q2_now))))                    
                )

        # 1. Compute generalized momentum
        # 3. Do semi-implicit integration.
        q1dot_next = q1dot_now + self.h * q1ddot
        q2dot_next = q2dot_now + self.h * q2ddot

        q1_next = torch.remainder(q1_now + self.h * q1dot_next, 2.0 * np.pi)
        q2_next = torch.remainder(q2_now + self.h * q2dot_next, 2.0 * np.pi)

        return torch.hstack((q1_next, q2_next, q1dot_next, q2dot_next))

    def dynamics_batch(self, x_batch, u_batch):
        """
        x_batch: torch.Tensor, shape: (B x n)
        u_batch: torch.Tensor, shape: (B x 1)
        """
        q1_now = x_batch[:,0].clone()
        q2_now = x_batch[:,1].clone()
        q1dot_now = x_batch[:,2].clone()
        q2dot_now = x_batch[:,3].clone()

        # Semi-implicit integration.

        # 1. Compute q2ddot

        q1ddot = (
            -self.g * (3 * self.m) * torch.sin(q1_now) - self.m * self.g * (
                torch.sin(q1_now - 2 * q2_now)) - 2 * torch.sin(
                    q1_now - q2_now) * (self.m * (
                        torch.pow(
                            q2dot_now, 2) * self.l + torch.pow(
                                q1dot_now, 2) * self.l * torch.cos(
                                    q1_now - q2_now)
                ))) / (
                    self.l * self.m * (3 - torch.cos(2 * (q1_now - q2_now))))

        q2ddot = (
            2 * torch.sin(q1_now - q2_now) * (torch.pow(
                q1_now, 2) * self.l * self.m * 2.0 + 2.0 * self.m * self.g * (
                    torch.cos(q1_now) + torch.pow(
                        q2dot_now, 2) * self.l * self.m * torch.cos(
                            q1_now - q2_now)
                )) / (
                    self.l * self.m * (3 - torch.cos(2 * (q1_now - q2_now))))                    
                )

        # 1. Compute generalized momentum
        # 3. Do semi-implicit integration.
        q1dot_next = q1dot_now + self.h * q1ddot
        q2dot_next = q2dot_now + self.h * q2ddot

        q1_next = torch.remainder(q1_now + self.h * q1dot_next, 2.0 * np.pi)
        q2_next = torch.remainder(q2_now + self.h * q2dot_next, 2.0 * np.pi)

        return torch.vstack((
            q1_next, q2_next, q1dot_next, q2dot_next)).transpose(0,1)

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

def test_pendulum_dynamics():
    dynamics = DoublePendulumDynamics()

    T = 4000

    x0 = torch.tensor([0.0, 3.0, 3.0, 0.0])
    u_trj = 5.0 * torch.ones(T, 1)
    x_trj = dynamics.rollout(x0, u_trj)

    x0_batch = torch.zeros(1000, 4)
    x0_batch[:,1] = 100.0
    u_trj_batch = 30.0 * torch.rand(1000, T, 1)
    x_trj_batch = dynamics.rollout_batch(x0_batch, u_trj_batch)

    plt.figure()
    plt.plot(x_trj[:,0])
    plt.plot(x_trj[:,1])
    plt.show()

#print(test_pendulum_dynamics())
