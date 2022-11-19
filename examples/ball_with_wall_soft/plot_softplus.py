import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import torch

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 20})

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm
from ball_with_wall_softplus import BallWithWallSoftPlus
from ball_with_wall import BallWithWallSoftPlusObjective

cm_f = cm.get_cmap("inferno")

plt.figure(figsize=(18,4))
colors = 'rgby'
kappa_space = np.array([1.0, 1.2, 1.4, 10.0])
B = 1000
plot_space = np.arange(0, B, 60)
theta_linspace = torch.linspace(0.0, np.pi/2, B)

for i in range(len(kappa_space)):
    dynamics = BallWithWallSoftPlus()
    dynamics.T = 250
    dynamics.kappa = np.power(10.0, kappa_space[i])
    objective = BallWithWallSoftPlusObjective(dynamics)
    objects = objective.evaluate_batch(theta_linspace, torch.zeros(B))
    plt.subplot(1,3,1)
    plt.plot(theta_linspace, objects, alpha=0.8,
        color=cm_f(1 - (i+1)/len(kappa_space)),
        label=r"$\log\kappa={kappa}$".format(kappa=kappa_space[i]))
    plt.title(r'$f_\kappa(\theta)$')
    plt.xlabel(r'$\theta$')
    matplotlib.rcParams.update({'font.size': 16})    
    plt.legend(loc='upper right')
    matplotlib.rcParams.update({'font.size': 20})    
    #plt.ylabel(r'$f_\kappa(\theta)$')



    plt.subplot(1,3,2)
    plt.title(r'f_\kappa(\theta)')
    true_grad = np.load(
        "results_ball/fobg_{:.2f}_true.npy".format(kappa_space[i]))
    plt.plot(theta_linspace, true_grad, alpha=0.8,
        color=cm_f(1 - (i+1)/len(kappa_space)))
    plt.title(r'$\nabla f_\kappa(\theta)$')
    plt.xlabel(r'$\theta$')
    #plt.ylabel(r'$f_\kappa(x)$')

    plt.subplot(1,3,3)
    plt.title('Trajectories')
    dynamics.render(plt.gca())
    trj_batch = dynamics.rollout_batch(theta_linspace)
    for b in plot_space:
        plt.plot(trj_batch[b,:,0],
            trj_batch[b,:,1], color=cm_f(1 - (i+1)/len(kappa_space)),
            alpha=0.8)
    plt.xlim([-0.2, 2])
    plt.ylim([-0.2, 2])

plt.savefig("trajectories")



matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(18,4))
for i in range(len(kappa_space)):
    fobg_storage = np.load(
        "results_ball/fobg_{:.2f}.npy".format(kappa_space[i]))
    zobg_storage = np.load(
        "results_ball/zobg_{:.2f}.npy".format(kappa_space[i]))
    fobg_var_storage = np.load(
        "results_ball/fobg_{:.2f}_var.npy".format(kappa_space[i]))
    zobg_var_storage = np.load(
        "results_ball/zobg_{:.2f}_var.npy".format(kappa_space[i]))

    plt.subplot(1,4,i+1)
    plt.plot(theta_linspace, zobg_var_storage / 10000, 'b-',
        label=r"Var[$\hat{\nabla}^0 F(\theta)]$")
    plt.plot(theta_linspace, fobg_var_storage / 10000, 'r-',
        label=r"Var[$\hat{\nabla}^1 F(\theta)$]")
    plt.xlabel(r"$\theta$")

    if (i == 0):
        plt.ylabel(r"Variance (10k)")
        plt.legend()
    plt.title(r"$\log\kappa={kappa}$".format(kappa=kappa_space[i]))
    plt.ylim([0, 6])
plt.savefig("kappa_variance.png")
plt.show()
