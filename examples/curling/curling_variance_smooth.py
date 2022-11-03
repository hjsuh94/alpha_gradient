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

num_kappaspace = 5
num_linspace = 100
kspace = np.linspace(2,7,num_linspace)
kappa_space = np.linspace(3,8,num_kappaspace)

fobg_var = np.zeros((num_kappaspace+1, num_linspace))
zobg_var = np.zeros((num_kappaspace+1, num_linspace))
stiffness_arr = np.zeros(num_linspace)

for k in range(len(kappa_space)):
    kappa = np.power(10.0, kappa_space[k])
    for i in tqdm(range(num_linspace)):
        stiffness = np.power(10.0, kspace[i])
        stiffness_arr[i] = stiffness
        damping = 0.6
        dynamics = CurlingDynamicsSmooth(stiffness, damping, kappa)
        dynamics.h = 0.005

        u_initial = 0.1 * torch.ones((T, dynamics.dim_u))
        u_initial = u_initial.reshape(T * dynamics.dim_u)

        objective = CurlingObjective(x0, xg, T, dynamics, Q, R, Qd)

        mean, var = objective.first_order_batch_gradient(u_initial, 100, 3e-1)
        fobg = mean
        fobg_var[k,i] = var
        mean, var = objective.zero_order_batch_gradient(u_initial, 100, 3e-1)
        zobg = mean
        zobg_var[k,i] = var

for i in tqdm(range(num_linspace)):
    stiffness = np.power(10.0, kspace[i])
    stiffness_arr[i] = stiffness
    damping = 0.6
    dynamics = CurlingDynamics(stiffness, damping)
    dynamics.h = 0.005

    u_initial = 0.1 * torch.ones((T, dynamics.dim_u))
    u_initial = u_initial.reshape(T * dynamics.dim_u)

    objective = CurlingObjective(x0, xg, T, dynamics, Q, R, Qd)

    mean, var = objective.first_order_batch_gradient(u_initial, 100, 3e-1)
    fobg = mean
    fobg_var[k+1,i] = var
    mean, var = objective.zero_order_batch_gradient(u_initial, 100, 3e-1)
    zobg = mean
    zobg_var[k+1,i] = var

np.save('curl_fobg_var.npy', fobg_var)
np.save('curl_zobg_var.npy', zobg_var)

plt.figure()
plt.plot(kspace, np.log(fobg_var), 'r-')
plt.plot(kspace, np.log(zobg_var), 'b-')
plt.show()
