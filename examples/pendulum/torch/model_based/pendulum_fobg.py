import numpy as np
import pydrake.symbolic as ps
import matplotlib.pyplot as plt
import time

import torch

from alpha_gradient.torch.fobgd_torch import FobgdTorch, FobgdTorchParams
from alpha_gradient.stepsize_scheduler import ManualScheduler
from pendulum_dynamics_torch import PendulumDynamicsTorch

system = PendulumDynamicsTorch(0.02)

T = 200
params = FobgdTorchParams()
params.gpu = False
params.Q = torch.diag(torch.Tensor([1,1]))
params.Qd = torch.diag(torch.Tensor([20, 20]))
params.R = torch.diag(torch.Tensor([1]))
params.x0 = torch.Tensor([0, 0])
params.xd_trj = torch.tile(torch.Tensor([np.pi, 0]), (T+1,1))
params.u_trj_initial = 0.1 * torch.ones((T, 1))

params.batch_size = 100
params.initial_std = 5.0 * torch.ones((T, 1))

def variance_scheduler(iter, initial_std):
    return initial_std / iter
params.variance_scheduler = variance_scheduler

def stepsize_schedule(iter, initial_stepsize):
    return initial_stepsize

stepsize_scheduler = ManualScheduler(stepsize_schedule, 0.01)
params.stepsize_scheduler = stepsize_scheduler

trajopt = FobgdTorch(system, params)
trajopt.iterate(30)

plt.figure()
plt.plot(trajopt.x_trj[:,0], trajopt.x_trj[:,1])
plt.show()

np.save("examples/pendulum/torch/analysis/fobg.npy", trajopt.x_trj)
