import numpy as np
import torch
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt 

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 27})

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.optimizer import (
    FobgdOptimizer, FobgdOptimizerParams,
    ZobgdOptimizer, ZobgdOptimizerParams,
    BiasConstrainedOptimizer, BiasConstrainedOptimizerParams)
from alpha_gradient.stepsize_scheduler import ManualScheduler    
from alpha_gradient.plotting_tools import plot_data, plot_data_log

from curling_objective import CurlingObjective
from curling_dynamics import CurlingDynamics

num_kappaspace = 1000
num_linspace = 100
kappa_space = -np.linspace(2,9,num_kappaspace)
kappa_expspace = np.power(10.0, kappa_space)

cost_fobg = np.load("fobg_var_smoothing_sweep.npy")
cost_zobg = np.load("zobg_var_smoothing_sweep.npy")

colormap_red = cm.get_cmap('Reds')
colormap_blue = cm.get_cmap('Blues')

plt.figure(figsize=(12,8))
plot_data(plt.gca(), kappa_space, np.log(cost_fobg[:-1]), 10,
        'red', label='FOBG (Smoothed)', fill_between=True)
plt.plot(kappa_space, np.log(cost_fobg[-1]) * np.ones(
    len(kappa_space)), color='red', linestyle='--', linewidth=2,
    label='FOBG (Non-smooth)')
plt.plot(kappa_space, (np.log(cost_fobg[-1]) + 2.6)* np.ones(
    len(kappa_space)), color='red', linestyle='dotted', alpha=0.5)
plt.plot(kappa_space, (np.log(cost_fobg[-1]) - 2.6)* np.ones(
    len(kappa_space)), color='red', linestyle='dotted', alpha=0.5)        


plot_data(plt.gca(), kappa_space, np.log(cost_zobg[:-1]), 10, 
        'blue', label='ZOBG (Smoothed)', fill_between=True)
plt.plot(kappa_space, np.log(cost_zobg[-1]) * np.ones(
    len(kappa_space)), color='blue', linestyle='--', linewidth=2,
    label='ZOBG (Non-smooth)')

plt.plot(kappa_space, (np.log(cost_zobg[-1]) + 0.04) * np.ones(
    len(kappa_space)), color='blue', linestyle='dotted', alpha=0.5)
plt.plot(kappa_space, (np.log(cost_zobg[-1]) - 0.04) * np.ones(
    len(kappa_space)), color='blue', linestyle='dotted', alpha=0.5)

plt.xlabel(r"Smoothing parameter $\kappa^{-1}$ (log-scale)")
plt.ylabel(r'Variance (log-scale)')
plt.legend()

plt.savefig('smoothing_variance_inverse.png')
plt.show()

