import os, shutil, subprocess

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem

class CurlingObjective(ObjectiveFunction):
    def __init__(self, x0, xg, T, dynamics, Q, R, Qd):
        super().__init__(T * dynamics.dim_u)
        self.x0 = x0
        self.xg = xg
        self.T = T 
        self.dynamics = dynamics

        self.d = T * dynamics.dim_u

        self.Q = Q
        self.R = R 
        self.Qd = Qd

    def evaluate_cost(self, x_trj, u_trj):
        """
        Evaluate cost given an state-input trajectory.
        - args:
            x_trj (shape (T + 1) x n): state trajectory
            u_trj (shape T x m): input trajectory
        """
        cost = 0.0

        for t in range(self.T):
            et = x_trj[t, :] - self.xg
            cost += (self.Q).mv(et).dot(et)
            cost += (self.R).mv(u_trj[t, :]).dot(u_trj[t, :])
        et = x_trj[self.T, :] - self.xg
        cost += (self.Qd).mv(et).dot(et)

        return cost

    def evaluate_cost_batch(self, x_trj, u_trj):
        """
        Evaluate cost given a batch of state-input trajectories.
        -args:
        x_trj (shape: B x (T+1) x n): state trajectory.
        u_trj (shape: B x T x m): input trajectory.
        """
        B = x_trj.shape[0]
        cost = torch.zeros(B)
        for t in range(self.T):
            et = x_trj[:, t, :] - self.xg
            cost += torch.diagonal(et.mm(self.Q).mm(et.transpose(0,1)))
            ut = u_trj[:,t,:]
            cost += torch.diagonal(ut.mm(self.R).mm(ut.transpose(0,1)))
        et = x_trj[:, self.T, :] - self.xg
        cost += torch.diagonal(et.mm(self.Qd).mm(et.transpose(0,1)))
        return cost

    def evaluate(self, x, w):
        """
        x: input array of shape (T x m,) (has to be one-dimensional)
        """
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        z = torch.tensor(x + w, dtype=torch.float32)
        u_trj = z.reshape(self.T, self.dynamics.dim_u) # reshape into trajectory\

        x_trj = self.dynamics.rollout(self.x0, u_trj)
        cost = self.evaluate_cost(x_trj, u_trj)
        return cost.detach().numpy()


    def evaluate_batch(self, x, w):
        """
        x: input array of shape (T x m,) (has to be one-dimensional)
        w: array of shape (B, T x m)
        """
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        B = w.shape[0]
        T = w.shape[1]
        z = torch.tensor(x + w, dtype=torch.float32)
        u_trj = z.reshape(B, self.T, self.dynamics.dim_u) # reshape into trajectory

        x_trj = self.dynamics.rollout_batch(self.x0, u_trj)
        cost = self.evaluate_cost_batch(x_trj, u_trj)
        return cost.detach().numpy()

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        z = torch.tensor(x + w, requires_grad=True, dtype=torch.float32)
        u_trj = z.reshape(self.T, self.dynamics.dim_u) # reshape into trajectory

        x_trj = self.dynamics.rollout(self.x0, u_trj)
        cost = self.evaluate_cost(x_trj, u_trj)
        cost.backward()
        return z.grad.detach().numpy()

    def gradient_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)        

        B = w.shape[0]
        z = torch.tensor(x + w, requires_grad=True, dtype=torch.float32)
        u_trj = z.reshape(B, self.T, self.dynamics.dim_u) # reshape into trajectory

        x_trj = self.dynamics.rollout_batch(self.x0, u_trj)
        cost = torch.sum(self.evaluate_cost_batch(x_trj, u_trj))
        cost.backward()
        return z.grad.detach().numpy()
