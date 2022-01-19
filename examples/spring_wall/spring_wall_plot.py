import numpy as np
import torch
import matplotlib.pyplot as plt 

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.optimizer import (
    FobgdOptimizer, FobgdOptimizerParams,
    ZobgdOptimizer, ZobgdOptimizerParams,
    BiasConstrainedOptimizer, BiasConstrainedOptimizerParams)
from alpha_gradient.stepsize_scheduler import ManualScheduler    
from alpha_gradient.plotting_tools import plot_cost

from spring_wall_objective import SpringWallObjective
from spring_wall_dynamics import SpringWallDynamics

cost_fobg = np.load("data/sw/sw_fobg_10_cost.npy")
cost_zobg = np.load("data/sw/sw_zobg_10_cost.npy")
cost_aobg = np.load("data/sw/sw_aobg_10_cost.npy")

plt.figure()
plt.subplot(1,2,1)
plt.title('k=10')
plot_cost(plt.gca(), np.log(cost_fobg), 20, 'red', label='FOBG')
plot_cost(plt.gca(), np.log(cost_zobg), 20, 'blue', label='ZOBG')
plot_cost(plt.gca(), np.log(cost_aobg), 20, 'springgreen', label='AOBG')
plt.xlabel('iterations')
plt.ylabel('cost')
plt.legend()

# load policy parameters.
cost_fobg = np.load("data/sw/sw_fobg_2000_cost.npy")
cost_zobg = np.load("data/sw/sw_zobg_2000_cost.npy")
cost_aobg = np.load("data/sw/sw_aobg_2000_cost.npy")



plt.subplot(1,2,2)
plt.title('k=2000')
plot_cost(plt.gca(), np.log(cost_fobg), 20, 'red', label='FOBG')
plot_cost(plt.gca(), np.log(cost_zobg), 20, 'blue', label='ZOBG')
plot_cost(plt.gca(), np.log(cost_aobg), 20, 'springgreen', label='AOBG')
plt.xlabel('iterations')
plt.ylabel('cost')
plt.legend()
plt.show()
