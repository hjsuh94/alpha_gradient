import numpy as np
import torch 
import matplotlib.pyplot as plt 
import os

from alpha_gradient.objective_function_policy import ObjectiveFunctionPolicy
from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.optimizer_policy import (
    FobgdPolicyOptimizer, FobgdPolicyOptimizerParams,
    ZobgdPolicyOptimizer, ZobgdPolicyOptimizerParams,
    BCPolicyOptimizer, BCPolicyOptimizerParams)
from alpha_gradient.stepsize_scheduler import ManualScheduler   
from alpha_gradient.policy import LinearPolicy 

from breakout_dynamics_toi import BreakoutDynamics
from breakout_policyopt import BreakoutPolicyOpt
from initial_distribution import sample_x0_batch, sample_x0_batch_narrow

kappa_array = np.array([1.9 1.5, 2, 2.5, 3, 3.5, 4])
data_folder = ".."

plt.figure()
for i in range(len(kappa_array)):
    fobg_filename = "fobg_softplus_1.5_cost.npy"
    fobg_cost = np.load(fobg_filename)
    plt.plot(np.log(fobg_cost), 'r-', alpha=kappa_array[i] / 4.0)
plt.show()
