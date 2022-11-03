import numpy as np
import torch
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt 

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 22})

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

num_kappaspace = 3
num_linspace = 100
k_space = np.linspace(2,7,num_linspace)
kappa_space = np.linspace(3,8,num_kappaspace)

k_expspace = np.power(10.0, k_space)
kappa_expspace = np.power(10.0, kappa_space)

cost_fobg = np.load("curl_fobg_var.npy")
cost_zobg = np.load("curl_zobg_var.npy")

colormap_red = cm.get_cmap('Reds')
colormap_blue = cm.get_cmap('Blues')

plt.figure()
for k in range(len(kappa_space)):
    plot_data(plt.gca(), 
        k_space, np.log(cost_fobg[k,:]), 7,
        colormap_red((k / len(kappa_space))),
        label='FOBG', fill_between=False)
    plot_data(plt.gca(),
        k_space, np.log(cost_zobg[k,:]), 7,
        colormap_blue((k / len(kappa_space))),
        label='ZOBG', fill_between=False)
plot_data(plt.gca(), 
    k_space, np.log(cost_fobg[k+1,:]), 7, 'red', label='FOBG',
        fill_between=False)
plot_data(plt.gca(),
    k_space, np.log(cost_zobg[k+1,:]), 7, 'blue', label='ZOBG',
        fill_between=False)
plt.xlabel("$km/c^2$ (log-scale)")
plt.ylabel('Variance (log-scale)')
plt.legend()
plt.show()

