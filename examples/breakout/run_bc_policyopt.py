import numpy as np
import torch 
import matplotlib.pyplot as plt 

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

# Set up dynamics.
dynamics = BreakoutDynamics()
sample_size = 1000
stdev = 0.01

# Initial condition.
xg = torch.tensor([0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
T = 200
Q = torch.diag(torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))
Qd = 100.0 * torch.diag(torch.tensor([1, 1, 0.1, 0.1, 0, 0, 0], dtype=torch.float32))
R = 0.001 * torch.diag(torch.tensor([1, 1, 1], dtype=torch.float32))

# Set up policy.
policy = LinearPolicy(dynamics.dim_x, dynamics.dim_u)
theta0 = torch.zeros(policy.d)

# Set up Objective.
objective = BreakoutPolicyOpt(T, dynamics, policy, Q, Qd, R, xg,
    sample_x0_batch)

#print(objective.zero_order_batch_gradient(theta0, sample_size, 0.01))
#print(objective.first_order_batch_gradient(theta0, sample_size, 0.01))

#============================================================================
params = BCPolicyOptimizerParams()
params.stdev = stdev
params.sample_size = sample_size
def constant_step(iter, initial_step): return 1e-6 * 1/(iter ** 0.1)
params.step_size_scheduler = ManualScheduler(constant_step, 1e-6)
params.theta0 = theta0
params.filename = "bc_narrow"
num_iters = 200

params.delta = 0.95
params.L = 1000
params.gamma = 1000

optimizer = BCPolicyOptimizer(objective, params)
optimizer.iterate(num_iters)

x_trj_batch, _ = objective.rollout_policy_batch(
    sample_x0_batch(100), torch.zeros(100, T, objective.m),
    optimizer.theta)

plt.figure()
for b in range(x_trj_batch.shape[0]):
    plt.plot(x_trj_batch[b,:,0], x_trj_batch[b,:,1])
plt.show()

plt.figure()
plt.plot(optimizer.cost_lst)
plt.show()


