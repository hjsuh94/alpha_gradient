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

from robot_motion import RobotMotion
from robot_dynamics import RoombaDynamics
from robot_map import RobotMap

# Set up map.
centers = torch.tensor([
    [-0.2,0.5,0.0,0.4, -0.8, -1, 0.0], 
    [0.5, 1.2, 2.3, 3.2, 3.8, 1., 1.5]
]).transpose(0,1)
radius = torch.tensor([0.2, 0.2, 0.3, 0.5, 0.1, 0.5, 0.2])
map = RobotMap(centers, radius)
plt.figure()
map.plot_map(plt.gca())

# Set up radius.
dynamics = RoombaDynamics(map)

# Initial condition.
x0 = torch.tensor([0,0])
xg = torch.Tensor([0.0,4.0])
T = 200
Q = torch.eye(2)
R = 0.001 * torch.eye(2)
Qd = 10.0 * torch.eye(2)

plt.plot(0.0, 4.0, 'x', color='purple', markersize=10, label='goal',
    alpha=0.9)

# Set up initial u.
u_initial = 0.01 * np.ones((T,2))
u_initial = u_initial.reshape(T*2)

objective = RobotMotion(x0, xg, T, dynamics, map, Q, R, Qd)

iters = 500

#============================================================================
params = BiasConstrainedOptimizerParams()
params.stdev = 0.05
params.sample_size = 500
def constant_step(iter, initial_step): return 0.01
params.step_size_scheduler = ManualScheduler(constant_step, 0.01)
params.x0_initial = u_initial

params.delta = 0.95
params.L = 10
params.gamma = 150.0
params.filename = "mp_bc"
num_iters = iters

optimizer = BiasConstrainedOptimizer(objective, params)
optimizer.iterate(num_iters)

u_trj_star = optimizer.x.reshape(T, dynamics.dim_u)
x_trj_star = dynamics.rollout(x0, u_trj_star)

plt.plot(x_trj_star[:,0], x_trj_star[:,1], '-', color='springgreen', label='AoBG')
#============================================================================
params = ZobgdOptimizerParams()
params.stdev = 0.05
params.sample_size = 500
def constant_step(iter, initial_step): return 0.01
params.step_size_scheduler = ManualScheduler(constant_step, 0.01)
params.x0_initial = u_initial
params.filename = "mp_zobg"
num_iters = iters

optimizer = ZobgdOptimizer(objective, params)
optimizer.iterate(num_iters)

u_trj_star = optimizer.x.reshape(T, dynamics.dim_u)
x_trj_star = dynamics.rollout(x0, u_trj_star)

plt.plot(x_trj_star[:,0], x_trj_star[:,1], 'b-', label='ZoBG')
#============================================================================
params = FobgdOptimizerParams()
params.stdev = 0.05
params.sample_size = 500
def constant_step(iter, initial_step): return 0.01
params.step_size_scheduler = ManualScheduler(constant_step, 0.01)
params.x0_initial = u_initial
params.filename = "mp_fobg"
num_iters = iters

optimizer = FobgdOptimizer(objective, params)
optimizer.iterate(num_iters)

u_trj_star = optimizer.x.reshape(T, dynamics.dim_u)
x_trj_star = dynamics.rollout(x0, u_trj_star)

plt.plot(x_trj_star[:,0], x_trj_star[:,1], 'r-', label='FoBG')
plt.axis('equal')
plt.legend()
plt.show()