import numpy as np
import pydrake.symbolic as ps
import time

import torch
import torch.nn as nn

from alpha_gradient.torch.dynamical_system_torch import DynamicalSystemTorch

class DynamicsNLP(nn.Module):
    def __init__(self):
        super(DynamicsNLP, self).__init__()

        self.dynamics_mlp = nn.Sequential(
            nn.Linear(3, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        return self.dynamics_mlp(x)

class PendulumDynamicsNN(DynamicalSystemTorch):
    def __init__(self, h, model_file_name):
        super().__init__()
        """
        x = [q, qdot]
        u = [tau]
        """
        self.h = h
        self.dim_x = 2
        self.dim_u = 1
        self.dynamics_net = DynamicsNLP()
        self.dynamics_net.load_state_dict(torch.load(model_file_name))
        self.dynamics_net.eval()

    def dynamics(self, x, u):
        """
        Numeric expression for dynamics.
        x (np.array, dim: n): state
        u (np.array, dim: m): action
        """

        xu = torch.Tensor(torch.hstack((x,u)))
        xnext = self.dynamics_net(xu)
        return xnext

    def dynamics_batch(self, x, u):
        """
        Batch dynamics. Uses pytorch for 
        -args:
            x (np.array, dim: B x n): batched state
            u (np.array, dim: B x m): batched input
        -returns:
            xnext (np.array, dim: B x n): batched next state
        """
        xu = torch.Tensor(torch.hstack((x,u)))
        xnext = self.dynamics_net(xu)
        return xnext

