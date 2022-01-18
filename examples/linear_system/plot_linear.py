import numpy as np
import torch 
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 19})

from alpha_gradient.objective_function_policy import ObjectiveFunctionPolicy
from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.policy import LinearPolicy
from alpha_gradient.optimizer_policy import (
    FobgdPolicyOptimizer, FobgdPolicyOptimizerParams,
    ZobgdPolicyOptimizer, ZobgdPolicyOptimizerParams,
    BCPolicyOptimizer, BCPolicyOptimizerParams)
from alpha_gradient.stepsize_scheduler import ManualScheduler       

from linear_dynamics import LinearDynamics
from linear_objective import LinearPolicyOpt

cost_zero = np.vstack((
    np.load("data/linear/linear_zero1_cost.npy"),
    np.load("data/linear/linear_zero2_cost.npy"),
    np.load("data/linear/linear_zero3_cost.npy"),
    np.load("data/linear/linear_zero4_cost.npy"),
    np.load("data/linear/linear_zero5_cost.npy")))                

cost_first = np.vstack((
    np.load("data/linear/linear_first1_cost.npy"),
    np.load("data/linear/linear_first2_cost.npy"),
    np.load("data/linear/linear_first3_cost.npy"),
    np.load("data/linear/linear_first4_cost.npy"),
    np.load("data/linear/linear_first5_cost.npy")))

cost_bc = np.vstack((
    np.load("data/linear/linear_bc1_cost.npy"),
    np.load("data/linear/linear_bc2_cost.npy"),
    np.load("data/linear/linear_bc3_cost.npy"),
    np.load("data/linear/linear_bc4_cost.npy"),
    np.load("data/linear/linear_bc5_cost.npy")))    

zero_mean = np.mean(cost_zero, axis=0)
zero_std = np.std(cost_zero, axis=0)
first_mean = np.mean(cost_first, axis=0)
first_std = np.std(cost_first, axis=0)
bc_mean = np.mean(cost_bc, axis=0)
bc_std = np.std(cost_bc, axis=0)

plt.figure()
plt.plot(np.log(zero_mean), 'b-', label="ZoBG")
plt.plot(np.log(first_mean), 'r-', label="FoBG")
plt.plot(np.log(bc_mean), '-', color='springgreen', label="AoBG")

plt.fill_between(range(len(zero_mean)),
    np.log(zero_mean - zero_std), np.log(zero_mean + zero_std),
    color='b', alpha=0.1)
plt.fill_between(range(len(first_mean)),
    np.log(first_mean - first_std), np.log(first_mean + first_std),
    color='b', alpha=0.1)    

plt.fill_between(range(len(bc_mean)),
    np.log(bc_mean - bc_std), np.log(bc_mean + bc_std),
    color='springgreen', alpha=0.1)
plt.fill_between(range(len(first_mean)),
    np.log(bc_mean - bc_std), np.log(bc_mean + bc_std),
    color='springgreen', alpha=0.1)    

plt.legend()
plt.ylabel('Cost (log-scale)')
plt.xlabel('Iterations')
plt.show()