import os, shutil, subprocess

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from alpha_gradient.objective_function_policy import ObjectiveFunctionPolicy
from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.policy import Policy

class BreakoutPolicyOpt(ObjectiveFunctionPolicy):
    """
    Policy optimization.
    parameters:
    - H: horizon of the problem.
    - dynamics: DynamicalSystem class.
    - policy: Policy class.
    - Q: quadratic cost on state (tensor.shape n x n)
    - Qd: quadratic cost on final state (tensor.shape n x n)
    - R: quadratic cost on input (tensor.shape m x m)
    - xg: goal configuration
    - sample_x0_batch: function that samples initial states.
        should have signature f(sample_size).
    """
    def __init__(self, H,
        dynamics: DynamicalSystem, policy: Policy, 
        Q, Qd, R, xg, sample_x0_batch):
        super().__init__(H, dynamics, policy)

        self.H = H
        self.xg = xg
        self.Q = Q
        self.Qd = Qd
        self.R = R

        self.sample_x0_batch = sample_x0_batch

    def evaluate_cost(self, x_trj, u_trj):
        """
        Evaluate cost given an state-input trajectory.
        - args:
            x_trj (shape (T + 1) x n): state trajectory
            u_trj (shape T x m): input trajectory
        """
        cost = 0.0
        for t in range(self.H):
            et = x_trj[t, :] - self.xg
            cost += (self.Q).mv(et).dot(et)
            cost += (self.R).mv(u_trj[t, :]).dot(u_trj[t, :])
        et = x_trj[self.H, :] - self.xg
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
        for t in range(self.H):
            et = x_trj[:, t, :] - self.xg
            cost += torch.diagonal(et.mm(self.Q).mm(et.transpose(0,1)))
            ut = u_trj[:,t,:]
            cost += torch.diagonal(ut.mm(self.R).mm(ut.transpose(0,1)))
        et = x_trj[:, self.H, :] - self.xg
        cost += torch.diagonal(et.mm(self.Qd).mm(et.transpose(0,1)))
        return cost
