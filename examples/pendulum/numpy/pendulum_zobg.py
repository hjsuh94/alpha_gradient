import numpy as np
from numpy.core.fromnumeric import var
import pydrake.symbolic as ps
import matplotlib.pyplot as plt
import time

from alpha_gradient.numpy.zobgd_np import ZobgdNp, ZobgdNpParams
from pendulum_dynamics_np import PendulumDynamicsNp

system = PendulumDynamicsNp(0.02)

T = 200
params = ZobgdNpParams()
params.Q = np.diag([1,1])
params.Qd = np.diag([20, 20])
params.R = np.diag([1])
params.x0 = np.array([0, 0])
params.xd_trj = np.tile(np.array([np.pi, 0]), (T+1,1))
params.u_trj_initial = 0.1 * np.ones((T, 1))

params.step_size = 0.015
params.batch_size = 100
params.initial_std = 0.1 * np.ones((T, 1))

def variance_scheduler(iter, initial_std):
    return initial_std / iter

def stepsize_scheduler(iter, initial_stepsize):
    return initial_stepsize / (iter ** 0.3)


params.variance_scheduler = variance_scheduler
params.stepsize_scheduler = stepsize_scheduler

trajopt = ZobgdNp(system, params)
trajopt.iterate(30)

plt.figure()
plt.plot(trajopt.x_trj[:,0], trajopt.x_trj[:,1])
plt.show()

np.save("examples/pendulum/numpy/analysis/zobgd.npy", trajopt.x_trj)