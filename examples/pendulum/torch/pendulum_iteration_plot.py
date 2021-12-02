import numpy as np
import pydrake.symbolic as ps
import matplotlib.pyplot as plt
import time

import torch

from alpha_gradient.torch.cem_torch import CemTorch, CemTorchParams
from alpha_gradient.torch.fobgd_torch import FobgdTorch, FobgdTorchParams
from alpha_gradient.torch.zobgd_torch import ZobgdTorch, ZobgdTorchParams
from alpha_gradient.stepsize_scheduler import ArmijoGoldsteinLineSearchTorch
from pendulum_dynamics_torch import PendulumDynamicsTorch

system = PendulumDynamicsTorch(0.02)

def run_cem(batch_size, iteration):
    T = 200
    params = CemTorchParams()
    params.gpu = False
    params.Q = torch.diag(torch.Tensor([1,1]))
    params.Qd = torch.diag(torch.Tensor([20, 20]))
    params.R = torch.diag(torch.Tensor([1]))
    params.x0 = torch.Tensor([0, 0])
    params.xd_trj = torch.tile(torch.Tensor([np.pi, 0]), (T+1,1))
    params.u_trj_initial = 0.1 * torch.ones((T, 1))

    params.n_elite = int(0.2 * (batch_size))
    params.batch_size = batch_size
    params.initial_std = torch.tile(torch.Tensor([2.0]), (T,1))

    trajopt = CemTorch(system, params)
    trajopt.iterate(iteration)

    return trajopt.cost_lst, trajopt.x_trj

def run_fobg(batch_size, iteration):
    T = 200
    params = FobgdTorchParams()
    params.gpu = False
    params.Q = torch.diag(torch.Tensor([1,1]))
    params.Qd = torch.diag(torch.Tensor([20, 20]))
    params.R = torch.diag(torch.Tensor([1]))
    params.x0 = torch.Tensor([0, 0])
    params.xd_trj = torch.tile(torch.Tensor([np.pi, 0]), (T+1,1))
    params.u_trj_initial = 0.1 * torch.ones((T, 1))

    params.batch_size = batch_size
    params.initial_std = 0.1 * torch.ones((T, 1))

    def variance_scheduler(iter, initial_std):
        return initial_std / iter
    params.variance_scheduler = variance_scheduler

    stepsize_scheduler = ArmijoGoldsteinLineSearchTorch(0.1, 0.1, 0.1)
    params.stepsize_scheduler = stepsize_scheduler

    trajopt = FobgdTorch(system, params)
    trajopt.iterate(iteration)

    return trajopt.cost_lst, trajopt.x_trj

def run_zobg(batch_size, iteration):
    T = 200
    params = ZobgdTorchParams()
    params.gpu = False
    params.Q = torch.diag(torch.Tensor([1,1]))
    params.Qd = torch.diag(torch.Tensor([20, 20]))
    params.R = torch.diag(torch.Tensor([1]))
    params.x0 = torch.Tensor([0, 0])
    params.xd_trj = torch.tile(torch.Tensor([np.pi, 0]), (T+1,1))
    params.u_trj_initial = 0.1 * torch.ones((T, 1))

    params.batch_size = batch_size
    params.initial_std = 0.1 * torch.ones((T, 1))

    def variance_scheduler(iter, initial_std):
        return initial_std / iter
    params.variance_scheduler = variance_scheduler

    stepsize_scheduler = ArmijoGoldsteinLineSearchTorch(0.1, 0.1, 0.1)
    params.stepsize_scheduler = stepsize_scheduler

    trajopt = ZobgdTorch(system, params)
    trajopt.iterate(iteration)

    return trajopt.cost_lst, trajopt.x_trj

def generate_plots(batch_size, iterations):
    cem_cost_lst, cem_trj = run_cem(batch_size, iterations)
    fobg_cost_lst, fobg_trj = run_fobg(batch_size, iterations)
    zobg_cost_lst, zobg_trj = run_zobg(batch_size, iterations)

    plt.figure()
    plt.plot(cem_cost_lst, '*-', color='royalblue', label='cem')
    plt.plot(fobg_cost_lst, 'r*-', label='fobg')
    plt.plot(zobg_cost_lst, '*-', color='springgreen', label='zobg')
    plt.title('Samples: {:02d}'.format(batch_size))
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(cem_trj[:,0], cem_trj[:,1], '-', color='royalblue', label='cem')
    plt.plot(fobg_trj[:,0], fobg_trj[:,1], '-', color='red', label='fobg')
    plt.plot(zobg_trj[:,0], zobg_trj[:,1], '-', color='springgreen', label='zobg')
    plt.title('Samples: {:02d}'.format(batch_size))    
    plt.xlabel('theta')
    plt.ylabel('thetadot')
    plt.legend()
    plt.show()

generate_plots(200, 20)

