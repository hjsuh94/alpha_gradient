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

from breakout_dynamics_toi import BreakoutDynamics
from breakout_trajopt import BreakoutTrajopt

# Set up dynamics.
dynamics = BreakoutDynamics()

# Initial condition.
x0 = torch.tensor([1.0, 2.0, -0.1, -0.2, 0.1, 0.0, -0.5], dtype=torch.float32)
xg = torch.tensor([0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
T = 400
Q = torch.diag(torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))
Qd = 100.0 * torch.diag(torch.tensor([1, 1, 0.1, 0.1, 0, 0, 0], dtype=torch.float32))
R = 0.001 * torch.diag(torch.tensor([1, 1, 1], dtype=torch.float32))

# Set up initial u.
u_initial = torch.zeros(T,3)
u_initial[:,0] = 0.0
u_initial[:,1] = 0.0
x_trj_initial = dynamics.rollout(x0, u_initial)
u_initial = u_initial.reshape(T*3)
objective = BreakoutTrajopt(x0, xg, T, dynamics, Q, Qd, R)

#dynamics.render_traj(x_trj_initial, xg)

"""
plt.figure()
plt.plot(x_trj_initial[:,0], x_trj_initial[:,1], 'r-')
plt.show()
"""

#============================================================================
params = FobgdOptimizerParams()
params.stdev = 0.1
params.sample_size = 100
def constant_step(iter, initial_step): return 0.0001
params.step_size_scheduler = ManualScheduler(constant_step, 0.0001)
params.x0_initial = u_initial
num_iters = 50

optimizer = FobgdOptimizer(objective, params)
optimizer.iterate(num_iters)

np.savetxt("gradient.csv",
    objective.gradient(
        optimizer.x, np.zeros(len(u_initial))).reshape(T, dynamics.dim_u),
    delimiter=",")

u_trj_star = optimizer.x.reshape(T, dynamics.dim_u)

x_trj_star = dynamics.rollout(x0, u_trj_star.float())

dynamics.render_traj(x_trj_star, xg)