import numpy as np
import pydrake.symbolic as ps
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from alpha_gradient.torch.dynamical_system_torch import DynamicalSystemTorch
from pendulum_dynamics_np import PendulumDynamicsNp
from pendulum_dynamics_torch import DynamicsNLP

"""1. Collect artificial data"""
pendulum = PendulumDynamicsNp(0.02)
dynamics = pendulum.dynamics
dynamics_batch = pendulum.dynamics_batch

num_data = 200000
xu = np.random.rand(num_data, 3)
xu[:,0] = 6 * np.pi * (xu[:,0] - 0.5)
xu[:,1] = 30.0 * (xu[:,1] - 0.5)
xu[:,2] = 30.0 * (xu[:,2] - 0.5)

xtarget = dynamics_batch(xu[:,0:2], xu[:,2,None])

"""3. Train the network."""
dynamics_net = DynamicsNLP()
dynamics_net.train()
optimizer = optim.Adam(dynamics_net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500)
criterion = nn.MSELoss()

num_iter = 600
for iter in tqdm(range(num_iter)):
    optimizer.zero_grad()
    output = dynamics_net(torch.Tensor(xu))
    loss = criterion(output, torch.Tensor(xtarget))
    loss.backward()
    optimizer.step()
    scheduler.step()

torch.save(dynamics_net.state_dict(),
    "examples/pendulum/torch/learned/weights/pendulum_weight.pth")