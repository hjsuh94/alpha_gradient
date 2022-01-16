import numpy as np
import torch 
import matplotlib.pyplot as plt

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

# Simple discrete-time linear dynamical system.

dynamics = LinearDynamics()
sample_size = 100
stdev = 0.01

# Initial conidtion.
xg = torch.tensor([0.0, 0.0], dtype=torch.float32)
T = 50
Q = torch.diag(torch.tensor([1.0, 1.0], dtype=torch.float32))
Qd = 10.0 * torch.diag(torch.tensor([1.0, 1.0], dtype=torch.float32))
R = 0.1 * torch.diag(torch.tensor([0.1, 0.1], dtype=torch.float32))

def sample_x0_batch(sample_size):
    return torch.normal(torch.ones(sample_size,2), 0.001 * torch.ones(sample_size,2))
    
print(sample_x0_batch(1000).shape)

# Sset up policy
policy = LinearPolicy(dynamics.dim_x, dynamics.dim_u)
theta0 = -0.1 * torch.tensor([1.0, 0.0, 0.1, 0.0])

# Set up objective
objective = LinearPolicyOpt(T, dynamics, policy, Q, Qd, R, xg, sample_x0_batch)

# Sanity check!
print(objective.zero_order_batch_gradient(theta0, sample_size, stdev))
print(objective.first_order_batch_gradient(theta0, sample_size, stdev))

params = ZobgdPolicyOptimizerParams()
params.stdev = stdev
params.sample_size = sample_size
def constant_step(iter, initial_step): return 1e-5
params.step_size_scheduler = ManualScheduler(constant_step, 1e-5)
params.theta0 = theta0
num_iters = 100

optimizer = ZobgdPolicyOptimizer(objective, params)
optimizer.iterate(num_iters)

print(optimizer.theta)

plt.figure()
plt.plot(optimizer.cost_lst)
plt.show()
