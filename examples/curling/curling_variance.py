import os, shutil, subprocess

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem

from curling_dynamics import CurlingDynamics
from curling_objective import CurlingObjective



x0 = torch.tensor([0.0, 1.0, 0.1, 0.0])
xg = torch.tensor([1.0, 2.0, 0.1, 0.0])
T = 100
Q = torch.diag(torch.tensor([0.1, 10.0, 0.1, 0.1]))
R = 1.0 * torch.eye(1)
Qd = 1.0 * Q

num_linspace = 10
kspace = np.linspace(1,4,num_linspace)

fobg_var = np.zeros(num_linspace)
zobg_var = np.zeros(num_linspace)
stiffness_arr = np.zeros(num_linspace)
condition = np.zeros(num_linspace)

for i in tqdm(range(num_linspace)):
    stiffness = np.power(10.0, kspace[i])
    stiffness_arr[i] = stiffness
    damping = 0.1
    dynamics = CurlingDynamics(stiffness, damping)
    dynamics.h = 0.005

    # Compute condition number.
    Amat = np.array([[0.0, 1.0], [-stiffness / 0.1, -damping/0.1]])
    eigval, _ = np.linalg.eig(Amat)
    eigmax = np.max(np.abs(eigval))
    eigmin = np.min(np.abs(eigval))

    print(eigval)

    cond = eigmax / eigmin
    condition[i] = cond

    print(cond)

    u_initial = 1.0 * torch.ones((T, dynamics.dim_u))
    u_initial = u_initial.reshape(T * dynamics.dim_u)

    objective = CurlingObjective(x0, xg, T, dynamics, Q, R, Qd)

    mean, var = objective.first_order_batch_gradient(u_initial, 10, 3e-1)
    fobg_var[i] = var
    mean, var = objective.zero_order_batch_gradient(u_initial, 10, 3e-1)
    zobg_var[i] = var

plt.figure()
x_trj = dynamics.rollout(x0, u_initial.reshape(T, dynamics.dim_u))
plt.plot(x_trj[:,0])
plt.show()

np.save('condition.npy', condition)
np.save('curl_fobg_var.npy', fobg_var)
np.save('curl_zobg_var.npy', zobg_var)

plt.figure()
plt.plot(kspace, np.log(fobg_var), 'r-')
plt.plot(kspace, np.log(zobg_var), 'b-')
plt.show()
