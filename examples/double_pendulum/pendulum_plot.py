import os, shutil, subprocess
from tqdm import tqdm

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 22})

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem

from double_pendulum_dynamics import DoublePendulumDynamics
from double_pendulum_objective import DoublePendulumObjective

from alpha_gradient.plotting_tools import plot_data

cost_fobg = np.load('data/dp2/dp_fobg_var.npy')
cost_zobg = np.load('data/dp2/dp_zobg_var.npy')

plt.figure()
plot_data(plt.gca(), range(len(cost_fobg)), np.log(cost_fobg), 5, 'red', label='FOBG')
plot_data(plt.gca(), range(len(cost_zobg)), np.log(cost_zobg), 5, 'blue', label='ZOBG')
plt.xlabel('Horizon (T)')
plt.ylabel('Variance (log-scale)')
plt.legend()
plt.show()
