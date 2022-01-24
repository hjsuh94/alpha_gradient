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

from curling_objective import CurlingObjective
from curling_dynamics import CurlingDynamics

stiffness = 1000
damping = 0.99
dynamics = CurlingDynamics(stiffness, damping)
dynamics.h = 0.005
step_size = 1.0 * 1e-3

# Initial condition.
x0 = torch.tensor([0.0, 1.0, 0.0, 0.0])
xg = torch.tensor([1.0, 2.0, 0.0, 0.0])
T = 200
Q = torch.diag(torch.tensor([0.1, 10.0, 0.1, 0.1]))
R = 1.0 * torch.eye(1)
Qd = 1.0 * Q

# Set up initial u
u_initial = 0.1 * np.ones((T,dynamics.dim_u))
x_initial = dynamics.rollout(x0, u_initial)
u_initial = u_initial.reshape(T*dynamics.dim_u)

objective = CurlingObjective(x0, xg, T, dynamics, Q, R, Qd)
iters = 500

"""
plt.figure()
plt.subplot(1,2,1)
plt.plot(range(T+1), x_initial[:,0], 'r-', label='ZoBG')
plt.plot(range(T+1), x_initial[:,1], 'b-', label='ZoBG')
plt.subplot(1,2,2)
plt.plot(range(T), u_initial, 'r-', label='ZoBG')
plt.show()
"""

#============================================================================
params = FobgdOptimizerParams()
params.stdev = 0.1
params.sample_size = 10
def constant_step(iter, initial_step): return step_size
params.step_size_scheduler = ManualScheduler(constant_step, step_size)
params.x0_initial = u_initial
params.filename = "curl_fobg_1000"
num_iters = iters

optimizer = FobgdOptimizer(objective, params)
optimizer.iterate(num_iters)

fobg_cost_array = optimizer.cost_lst

u_trj_star = optimizer.x.reshape(T, dynamics.dim_u)
x_trj_star = dynamics.rollout(x0, u_trj_star)

"""
plt.figure()
plt.subplot(1,2,1)
plt.plot(range(T+1), x_trj_star[:,0], 'r-', label='ZoBG')
plt.plot(range(T+1), x_trj_star[:,1], 'b-', label='ZoBG')
plt.subplot(1,2,2)
plt.plot(range(T), u_trj_star, 'r-', label='ZoBG')
plt.show()
"""

#============================================================================
params = BiasConstrainedOptimizerParams()
params.stdev = 0.1
params.sample_size = 10
def constant_step(iter, initial_step): return step_size
params.step_size_scheduler = ManualScheduler(constant_step, step_size)
params.x0_initial = u_initial
params.filename = "curl_aobg_1000"
num_iters = iters

params.delta = 0.95
params.L = 100.0
params.gamma = 1e8

optimizer = BiasConstrainedOptimizer(objective, params)
optimizer.iterate(num_iters)

aobg_cost_array = optimizer.cost_lst

u_trj_star = optimizer.x.reshape(T, dynamics.dim_u)
x_trj_star = dynamics.rollout(x0, u_trj_star)

"""
plt.figure()
plt.subplot(1,2,1)
plt.plot(range(T+1), x_trj_star[:,0], 'r-', label='ZoBG')
plt.plot(range(T+1), x_trj_star[:,1], 'b-', label='ZoBG')
plt.subplot(1,2,2)
plt.plot(range(T), u_trj_star, 'r-', label='ZoBG')
plt.show()
"""

#============================================================================
params = ZobgdOptimizerParams()
params.stdev = 0.1
params.sample_size = 10
def constant_step(iter, initial_step): return step_size
params.step_size_scheduler = ManualScheduler(constant_step, step_size)
params.x0_initial = u_initial
params.filename = "curl_zobg_1000"
num_iters = iters

optimizer = ZobgdOptimizer(objective, params)
optimizer.iterate(num_iters)

zobg_cost_array = optimizer.cost_lst

u_trj_star = optimizer.x.reshape(T, dynamics.dim_u)
x_trj_star = dynamics.rollout(x0, u_trj_star)

"""
plt.figure()
plt.subplot(1,2,1)
plt.plot(range(T+1), x_trj_star[:,0], 'r-', label='ZoBG')
plt.plot(range(T+1), x_trj_star[:,1], 'b-', label='ZoBG')
plt.subplot(1,2,2)
plt.plot(range(T), u_trj_star, 'r-', label='ZoBG')
plt.show()
"""



#============================================================================
plt.figure()
plt.plot(zobg_cost_array, 'b-')
plt.plot(fobg_cost_array, 'r-')
plt.plot(aobg_cost_array, '-', color='springgreen')
plt.show()


"""

"""