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

from robot_motion import RobotMotion, RobotMap
from robot_dynamics import RoombaDynamics, BicycleDynamics

# Set up map.
map = RobotMap()
dynamics = BicycleDynamics(map)

# Initial condition.
x0 = torch.tensor([0,0,0,0,0], dtype=torch.float32)
xg = torch.tensor([0.3,0.2,np.pi/2,0,0], dtype=torch.float32)
T = 100
Q = torch.diag(torch.tensor([1,1,1,0.1,0.1], dtype=torch.float32))
R = torch.diag(torch.tensor([0.5,0.5], dtype=torch.float32))
Qd = 1000 * Q

# Set up initial u.
u_initial = 0.05 * np.ones((T,2))
u_initial[:,1] = 0.0
u_initial = u_initial.reshape(T*2)

objective = RobotMotion(x0, xg, T, dynamics, map, Q, R, Qd)
print(objective.evaluate(u_initial, np.zeros(T*2)))
print(objective.gradient(u_initial, np.zeros(T*2)))

params = FobgdOptimizerParams()
params.stdev = 0.005
params.sample_size = 1000
def constant_step(iter, initial_step): return 0.001
params.step_size_scheduler = ManualScheduler(constant_step, 0.001)
params.x0_initial = u_initial
num_iters = 500

optimizer = FobgdOptimizer(objective, params)
optimizer.iterate(num_iters)

u_trj_star = optimizer.x.reshape(T, dynamics.dim_u)
x_trj_star = dynamics.rollout(x0, torch.tensor(u_trj_star))

plt.figure()
plt.plot(x_trj_star[:,0], x_trj_star[:,1], 'r-')
plt.show()