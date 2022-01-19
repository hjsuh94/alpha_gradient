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

from robot_motion import RobotMotion
from robot_dynamics import RoombaDynamics
from robot_map import RobotMap

cost_fobg = np.load("data/mp3/mp_fobg_cost.npy")
cost_zobg = np.load("data/mp3/mp_zobg_cost.npy")
cost_aobg = np.load("data/mp3/mp_bc_cost.npy")

plt.figure()
plot_cost(plt.gca(), cost_fobg, 10, 'red', label='FOBG')
plot_cost(plt.gca(), cost_zobg, 10, 'blue', label='ZOBG')
plot_cost(plt.gca(), cost_aobg, 10, 'springgreen', label='AOBG')
plt.xlabel('iterations')
plt.ylabel('cost')
plt.legend()
plt.show()