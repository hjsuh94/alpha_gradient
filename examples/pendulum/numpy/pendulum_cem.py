import numpy as np
import pydrake.symbolic as ps
import matplotlib.pyplot as plt
import time

from alpha_gradient.numpy.cem_np import CemNp, CemNpParams
from pendulum_dynamics_np import PendulumDynamicsNp

system = PendulumDynamicsNp(0.02)

T = 200
params = CemNpParams()
params.Q = np.diag([1,1])
params.Qd = np.diag([20, 20])
params.R = np.diag([1])
params.x0 = np.array([0, 0])
params.xd_trj = np.tile(np.array([np.pi, 0]), (T+1,1))
params.u_trj_initial = 0.1 * np.ones((T, 1))

params.n_elite = 20
params.batch_size = 100
params.initial_std = np.tile(2.0, (T,1))

trajopt = CemNp(system, params)
trajopt.iterate(30)

plt.figure()
plt.plot(trajopt.x_trj[:,0], trajopt.x_trj[:,1])
plt.show()

np.save("examples/pendulum/numpy/analysis/cem.npy", trajopt.x_trj)