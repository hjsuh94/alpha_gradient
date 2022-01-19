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
from alpha_gradient.plotting_tools import plot_cost

from curling_objective import CurlingObjective
from curling_dynamics import CurlingDynamics

cost_fobg = np.load("data/curling/curl_fobg_1000_cost.npy")
cost_zobg = np.load("data/curling/curl_zobg_1000_cost.npy")
cost_aobg = np.load("data/curling/curl_aobg_1000_cost.npy")

plt.figure()
plt.title('k=1000.0')
plot_cost(plt.gca(), np.log(cost_fobg), 20, 'red', label='FOBG')
plot_cost(plt.gca(), np.log(cost_zobg), 20, 'blue', label='ZOBG')
plot_cost(plt.gca(), np.log(cost_aobg), 20, 'springgreen', label='AOBG')
plt.xlabel('iterations')
plt.ylabel('cost (log-scale)')
plt.legend()

cost_fobg = np.load("data/curling2/curl_fobg_10_cost.npy")
cost_zobg = np.load("data/curling2/curl_zobg_10_cost.npy")
cost_aobg = np.load("data/curling2/curl_aobg_10_cost.npy")

plt.figure()
plt.title('k=10.0')
plot_cost(plt.gca(), np.log(cost_zobg), 20, 'blue', label='ZOBG')
plot_cost(plt.gca(), np.log(cost_aobg), 20, 'springgreen', label='AOBG')
plot_cost(plt.gca(), np.log(cost_fobg), 20, 'red', label='FOBG', style='--')
plt.xlabel('iterations')
plt.ylabel('cost (log-scale)')
plt.legend()
plt.show()
