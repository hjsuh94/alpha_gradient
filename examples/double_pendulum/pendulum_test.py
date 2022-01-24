import os, shutil, subprocess
from tqdm import tqdm

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem

from double_pendulum_dynamics import DoublePendulumDynamics
from double_pendulum_objective import DoublePendulumObjective


dynamics = DoublePendulumDynamics()
Q = torch.diag(torch.tensor([1.0, 1.0, 0.1, 0.1]))
T = 2000
Qd = T * torch.diag(torch.tensor([1.0, 1.0, 0.1, 0.1]))
xg = torch.tensor([np.pi, np.pi, 0.0, 0.0])


num_grid = 200
fobg_var = np.zeros(num_grid)
zobg_var = np.zeros(num_grid)

for t in tqdm(range(num_grid)):
    objective = DoublePendulumObjective(xg, 10 * t, dynamics, Q, Qd)

    mean, var = objective.first_order_batch_gradient(
            torch.tensor([np.pi/2, 0.0 ,5, -5]), 100, 0.5)
    fobg_var[t] = var

    mean, var = objective.zero_order_batch_gradient(
            torch.tensor([np.pi/2, 0.0 ,5, -5]), 100, 0.5)
    zobg_var[t] = var

np.save("dp_fobg_var.npy", fobg_var)
np.save("dp_zobg_var.npy", zobg_var)

plt.figure()
plt.plot(fobg_var, 'r-')
plt.plot(zobg_var, 'b-')
plt.show()
