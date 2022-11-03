import os, shutil, subprocess

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem

from curling_dynamics import CurlingDynamics
from curling_dynamics_smooth import CurlingDynamicsSmooth
from curling_objective import CurlingObjective

torch.set_default_tensor_type(torch.DoubleTensor)



x0 = torch.tensor([0.0, 1.0, 0.0, 0.0])
xg = torch.tensor([1.0, 2.0, 0.0, 0.0])
T = 30
Q = torch.diag(torch.tensor([0.1, 10.0, 0.1, 0.1]))
R = 1.0 * torch.eye(1)
Qd = 1.0 * Q

num_linspace = 1000
kspace = np.linspace(2,6,num_linspace)
kappa_space = np.linspace(2,9,num_linspace)
fobg_var = np.zeros(num_linspace)
zobg_var = np.zeros(num_linspace)
condition = np.zeros(num_linspace)
stiffness = np.power(10.0, 5)
damping = 0.6

"""
for i in tqdm(range(len(kappa_space))):
    kappa = np.power(10.0, kappa_space[i])
    dynamics = CurlingDynamicsSmooth(stiffness, damping, kappa)
    dynamics.h = 0.005

    u_initial = 0.1 * torch.ones((T, dynamics.dim_u))
    u_initial = u_initial.reshape(T * dynamics.dim_u)

    objective = CurlingObjective(x0, xg, T, dynamics, Q, R, Qd)

    mean, var = objective.first_order_batch_gradient(u_initial, 1000, 3e-1)
    fobg = mean
    fobg_var[i] = var
    mean, var = objective.zero_order_batch_gradient(u_initial, 1000, 3e-1)
    zobg = mean
    zobg_var[i] = var

    #np.save('curl_fobg_var_{}.npy'.format(kappa_space[k]), fobg_var)
    #np.save('curl_zobg_var_{}.npy'.format(kappa_space[k]), zobg_var)
"""

fobg_var_array = np.zeros(100)
zobg_var_array = np.zeros(100)
for k in range(100):
    dynamics = CurlingDynamics(stiffness, damping)
    dynamics.h = 0.005

    u_initial = 0.1 * torch.ones((T, dynamics.dim_u))
    u_initial = u_initial.reshape(T * dynamics.dim_u)

    objective = CurlingObjective(x0, xg, T, dynamics, Q, R, Qd)

    mean, var = objective.first_order_batch_gradient(u_initial, 1000, 3e-1)
    fobg = mean
    fobg_var_array[k] = var
    mean, var = objective.zero_order_batch_gradient(u_initial, 1000, 3e-1)
    zobg = mean
    zobg_var_array[k] = var

print(np.std(np.log(fobg_var_array)))
print(np.std(np.log(zobg_var_array)))


"""
plt.figure()
T = 100
dynamics = CurlingDynamics(10000, 0.01)
u_initial = 500.0 * torch.ones((T, dynamics.dim_u))
x_trj = dynamics.rollout(x0, u_initial.reshape(T, dynamics.dim_u))
plt.plot(x_trj[:,0])
plt.plot(x_trj[:,1])
plt.show()
"""

"""
np.save('fobg_var_smoothing_sweep.npy', np.hstack((fobg_var, true_fobg_var)))
np.save('zobg_var_smoothing_sweep.npy', np.hstack((zobg_var, true_zobg_var)))

plt.figure()
plt.plot(kappa_space, np.log(fobg_var), 'r-')
plt.plot(kappa_space, np.log(zobg_var), 'b-')
plt.show()
"""