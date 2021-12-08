import numpy as np
import pydrake.symbolic as ps
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 
import torch

from alpha_gradient.torch.fobgd_torch import FobgdTorch, FobgdTorchParams
from alpha_gradient.torch.zobgd_torch import ZobgdTorch, ZobgdTorchParams
from alpha_gradient.stepsize_scheduler import ArmijoGoldsteinLineSearchTorch
from pendulum_dynamics_torch import PendulumDynamicsTorch

from alpha_gradient.statistical_analysis import compute_variance

system = PendulumDynamicsTorch(0.02)

def run_fobg(batch_size, horizon, estimation_size):
    T = horizon
    params = FobgdTorchParams()
    params.gpu = False
    params.Q = torch.diag(torch.Tensor([1,1]))
    params.Qd = torch.diag(torch.Tensor([20, 20]))
    params.R = torch.diag(torch.Tensor([1]))
    params.x0 = torch.Tensor([0, 0])
    params.xd_trj = torch.tile(torch.Tensor([np.pi, 0]), (T+1,1))
    params.u_trj_initial = 0.1 * torch.ones((T, 1))

    params.batch_size = batch_size
    params.initial_std = 0.3 * torch.ones((T, 1))

    def variance_scheduler(iter, initial_std):
        return initial_std / iter
    params.variance_scheduler = variance_scheduler

    stepsize_scheduler = ArmijoGoldsteinLineSearchTorch(0.1, 0.1, 0.1)
    params.stepsize_scheduler = stepsize_scheduler

    trajopt = FobgdTorch(system, params)
    fobg_batch = np.zeros((estimation_size, T))
    for i in range(estimation_size):
        fobg_batch[i] = trajopt.compute_fobg(
            trajopt.x_trj, trajopt.u_trj).squeeze(1).detach().numpy()

    return compute_variance(fobg_batch, p='nuc')


def run_zobg(batch_size, horizon, estimation_size):
    T = horizon
    params = ZobgdTorchParams()
    params.gpu = False
    params.Q = torch.diag(torch.Tensor([1,1]))
    params.Qd = torch.diag(torch.Tensor([20, 20]))
    params.R = torch.diag(torch.Tensor([1]))
    params.x0 = torch.Tensor([0, 0])
    params.xd_trj = torch.tile(torch.Tensor([np.pi, 0]), (T+1,1))
    params.u_trj_initial = 0.1 * torch.ones((T, 1))

    params.batch_size = batch_size
    params.initial_std = 0.3 * torch.ones((T, 1))

    def variance_scheduler(iter, initial_std):
        return initial_std / iter
    params.variance_scheduler = variance_scheduler

    stepsize_scheduler = ArmijoGoldsteinLineSearchTorch(0.1, 0.1, 0.1)
    params.stepsize_scheduler = stepsize_scheduler

    trajopt = ZobgdTorch(system, params)
    zobg_batch = np.zeros((estimation_size, T))
    for i in range(estimation_size):
        zobg_batch[i] = trajopt.compute_zobg(
            trajopt.x_trj, trajopt.u_trj).squeeze(1).detach().numpy()

    return compute_variance(zobg_batch, p='nuc')    

def horizon_sweep(max_horizon, batch_size, estimation_size):
    fobg_variance = np.zeros(max_horizon)
    zobg_variance = np.zeros(max_horizon)
    for h in tqdm(range(1,max_horizon)):
        fobg_variance[h] = run_fobg(batch_size, h, estimation_size)
        zobg_variance[h] = run_zobg(batch_size, h, estimation_size)

    plt.figure()
    plt.plot(fobg_variance, '-', color='red', label='fobg variance')
    plt.plot(zobg_variance, '-', color='springgreen', label='zobg variance')
    plt.legend()
    plt.xlabel('problem horizon')
    plt.ylabel('variance of gradient estimate')
    plt.show()

def sample_sweep(horizon, max_batch_size, estimation_size):
    fobg_variance = np.zeros(max_batch_size-2)
    zobg_variance = np.zeros(max_batch_size-2)
    for b in tqdm(range(2,max_batch_size)):
        fobg_variance[b-2] = run_fobg(b, horizon, estimation_size)
        zobg_variance[b-2] = run_zobg(b, horizon, estimation_size)

    plt.figure()
    plt.plot(fobg_variance, '-', color='red', label='fobg variance')
    plt.plot(zobg_variance, '-', color='springgreen', label='zobg variance')
    plt.legend()
    plt.xlabel('number of samples')
    plt.ylabel('variance of gradient estimate')
    plt.show()    

#horizon_sweep(200, 50, 20)

sample_sweep(50, 200, 20)