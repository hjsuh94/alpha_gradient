import numpy as np
import torch
import matplotlib
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

stiffness = np.load('data/curling_var/stiffness.npy')
cost_fobg = np.load("data/curling_var/curl_fobg_var.npy")
cost_zobg = np.load("data/curling_var/curl_zobg_var.npy")

stiffness = np.log(np.exp(stiffness / 0.1))

plt.figure()
plot_data(plt.gca(), stiffness, np.log(cost_fobg), 3, 'red', label='FOBG')
plot_data(plt.gca(), stiffness, np.log(cost_zobg), 3, 'blue', label='ZOBG')
plt.xlabel("$km/c^2$ (log-scale)")
plt.ylabel('Variance (log-scale)')
plt.legend()
plt.show()

