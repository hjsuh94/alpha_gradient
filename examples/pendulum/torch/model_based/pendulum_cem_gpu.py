import numpy as np
import pydrake.symbolic as ps
import matplotlib.pyplot as plt
import time

import torch

from alpha_gradient.torch.cem_torch import CemTorch, CemTorchParams
from pendulum_dynamics_torch import PendulumDynamicsTorch

system = PendulumDynamicsTorch(0.02)

T = 200
params = CemTorchParams()
params.gpu = True
params.Q = torch.diag(torch.Tensor([1,1])).cuda()
params.Qd = torch.diag(torch.Tensor([20, 20])).cuda()
params.R = torch.diag(torch.Tensor([1])).cuda()
params.x0 = torch.Tensor([0, 0])
params.xd_trj = torch.tile(torch.Tensor([np.pi, 0]), (T+1,1)).cuda()
params.u_trj_initial = 0.1 * torch.ones((T, 1)).cuda()

params.n_elite = 20
params.batch_size = 100
params.initial_std = torch.tile(torch.Tensor([2.0]), (T,1)).cuda()

trajopt = CemTorch(system, params)
trajopt.iterate(30)

plt.figure()
plt.plot(trajopt.x_trj[:,0], trajopt.x_trj[:,1])
plt.show()

np.save("examples/pendulum/torch/analysis/cem.npy", trajopt.x_trj)
