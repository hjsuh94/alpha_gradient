import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from tqdm import tqdm

import torch

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm
from pivot_objective import PivotObjective
from pivot_dynamics import PivotSoftPlus


plt.figure(figsize=(16,10))
matplotlib.rcParams.update({'font.size': 22})
colors = ["magenta", "springgreen", "royalblue", "orangered"]
B = 1000
kappa_space = np.array([8,10,20,100.0])
plot_space = np.arange(0, B, 40)

dynamics = PivotSoftPlus()
dynamics.T = 100
dynamics.kappa = 100.0
objective = PivotObjective(dynamics)
theta_linspace = torch.linspace(0, np.pi/2, B)

for i in range(len(kappa_space)):
    dr_cost = 0.0
    for j in range(1000):
        dynamics.height = 0.5 + 0.2 * np.random.normal(0.0, 1.0)
        objective = PivotObjective(dynamics)    
        dr_cost += objective.evaluate_batch(theta_linspace, torch.zeros(B))
    objects_dr = dr_cost / 1000

    dynamics.height = 0.5
    objective = PivotObjective(dynamics)    
    objects = objective.evaluate_batch(theta_linspace, torch.zeros(B))
    plt.subplot(1,2,1)
    plt.plot(theta_linspace, objects, linestyle='--', color=colors[i],alpha=1.0, label=str(dynamics.kappa))
    plt.plot(theta_linspace, objects_dr, color=colors[i], alpha=0.5, label=str(dynamics.kappa))
    plt.ylabel("Cost")
    plt.xlabel(r'$\theta(rad)$')
    plt.legend()

    plt.subplot(1,2,2)
    # dynamics.render(plt.gca())

    vx = dynamics.v0 * torch.cos(theta_linspace)
    vy = dynamics.v0 * torch.sin(theta_linspace)
    x0 = torch.hstack([torch.zeros((B,3)), vx[:,None], vy[:,None], torch.zeros((B,1))])    
    trj_batch = dynamics.rollout_batch(x0)

    T = 90
    b0 = 300
    b1 = 500
    plt.plot(trj_batch[b0,:,0], trj_batch[b0,:,1], color=colors[i], alpha=0.8)
    plt.plot(trj_batch[b1,:,0], trj_batch[b1,:,1], color=colors[i], alpha=0.8)

    plt.plot(trj_batch[b0, T, 0], trj_batch[b0, T, 1], 'ro', color=colors[i], markersize=7)
    plt.plot(trj_batch[b1, T, 0], trj_batch[b1, T, 1], 'ro', color=colors[i], markersize=7)
    dynamics.render(plt.gca(), torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    plt.axis('equal')
    plt.xlabel(r'$x(m)$')
    plt.ylabel(r'$y(m)$')    

    fobg_storage = np.zeros((B, 1))
    zobg_storage = np.zeros((B, 1))
    fobg_var_storage = np.zeros((B, 1))
    zobg_var_storage = np.zeros((B, 1))    
    
plt.show()
