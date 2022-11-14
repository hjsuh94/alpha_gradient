import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import torch

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm
from ball_with_wall_softplus import BallWithWallSoftPlus
from ball_with_wall import BallWithWallSoftPlusObjective


plt.figure()
colors = 'rgby'
kappa_space = np.array([1.0, 1.2, 1.4, 10.0])
B = 1000
plot_space = np.arange(0, B, 40)

for i in range(len(kappa_space)):
    dynamics = BallWithWallSoftPlus()
    dynamics.T = 250
    dynamics.kappa = np.power(10.0, kappa_space[i])
    objective = BallWithWallSoftPlusObjective(dynamics)
    theta_linspace = torch.linspace(0.0, np.pi/2, B)
    objects = objective.evaluate_batch(theta_linspace, torch.zeros(B))
    plt.subplot(2,2,1)
    plt.plot(theta_linspace, objects, 'r', alpha=(i+1) / len(kappa_space))

    plt.subplot(2,2,2)
    dynamics.render(plt.gca())
    trj_batch = dynamics.rollout_batch(theta_linspace)
    for b in plot_space:
        plt.plot(trj_batch[b,:,0], trj_batch[b,:,1], colors[i])

    fobg_storage = np.zeros((B, 1))
    zobg_storage = np.zeros((B, 1))
    fobg_var_storage = np.zeros((B, 1))
    zobg_var_storage = np.zeros((B, 1))    

    for b in tqdm(range(B)):
        """
        mean, var = objective.zero_order_batch_gradient(
                    torch.tensor([theta_linspace[b]]), 1000, 0.05)
        zobg_storage[b] = mean
        zobg_var_storage[b] = var
        """
        mean, var = objective.first_order_batch_gradient(
            theta_linspace[b], 1, 0.0)
        fobg_storage[b] = mean
        fobg_var_storage[b] = var

    np.save("results_ball/fobg_{:.2f}_true.npy".format(kappa_space[i]),
        fobg_storage)
    """"
    np.save("results/fobg_{:.2f}_var.npy".format(kappa_space[i]),
        fobg_var_storage)
    np.save("results/zobg_{:.2f}.npy".format(kappa_space[i]),
        zobg_storage)
    np.save("results/zobg_{:.2f}_var.npy".format(kappa_space[i]),
        zobg_var_storage)        

    plt.subplot(2,2,3)
    plt.plot(theta_linspace, fobg_storage, 'r-', alpha= (i + 1)/ len(kappa_space))
    plt.plot(theta_linspace, zobg_storage, 'b-', alpha= (i+1) / len(kappa_space))
    
    plt.subplot(2,2,4)
    plt.plot(theta_linspace, fobg_var_storage, 'r-', alpha= (i + 1)/ len(kappa_space))
    plt.plot(theta_linspace, zobg_var_storage, 'b-', alpha= (i+1) / len(kappa_space))    
    """        
        
plt.show()
