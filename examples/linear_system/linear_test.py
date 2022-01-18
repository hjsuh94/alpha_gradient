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
d = 100
dynamics = LinearDynamics(d)
sample_size = 10
stdev = 0.1


# Initial conidtion.
xg = torch.zeros(d, dtype=torch.float32)
T = 10
Q = torch.diag(torch.ones(d, dtype=torch.float32))
Qd = 10.0 * torch.diag(torch.ones(d, dtype=torch.float32))
R = 0.1 * torch.diag(torch.ones(d, dtype=torch.float32))

def sample_x0_batch(sample_size):
    return torch.normal(
        torch.ones(sample_size,d), 0.001 * torch.ones(sample_size,d))
    
print(sample_x0_batch(1000).shape)

# Sset up policy
policy = LinearPolicy(dynamics.dim_x, dynamics.dim_u)
theta0 = -0.1 * torch.zeros(d * d)

# Set up objective
objective = LinearPolicyOpt(T, dynamics, policy, Q, Qd, R, xg, sample_x0_batch)

# Sanity check!
print(objective.zero_order_batch_gradient(theta0, sample_size, stdev))
print(objective.first_order_batch_gradient(theta0, sample_size, stdev))

params = BCPolicyOptimizerParams()
params.stdev = stdev
params.sample_size = sample_size
def constant_step(iter, initial_step): return 1e-7
params.step_size_scheduler = ManualScheduler(constant_step, 1e-7)
params.filename = "linear_bc5"
params.theta0 = theta0
num_iters = 100

params.delta = 0.95
params.L = 100
params.gamma = 100

optimizer = BCPolicyOptimizer(objective, params)
optimizer.iterate(num_iters)

print(optimizer.theta)

plt.figure()
plt.plot(optimizer.cost_lst)
plt.show()
