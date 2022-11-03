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

from breakout_dynamics_smooth import BreakoutDynamics
from breakout_trajopt import BreakoutTrajopt

# Set up dynamics.
dynamics = BreakoutDynamics()

# Initial condition.
x0 = torch.tensor([1.0, 2.0, -0.2, -0.4, 1.0, 0.0, -0.2], dtype=torch.float32)
xg = torch.tensor([0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
T = 200
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
coordinates = torch.rand(1000,2)
coordinates[:,0] = -1.5 + 3.0 * coordinates[:,0]
coordinates[:,1] = -1.5 + 3.0 * coordinates[:,1]

dists, normals = dynamics.compute_sg_batch(coordinates)

plt.figure()
#plt.plot(coordinates[:,0].numpy(), coordinates[:,1].numpy(), 'ro')

dists = torch.zeros(1000)
normals = torch.zeros(1000,2)

for i in range(1000):
    dist, normal = dynamics.compute_sg(coordinates[i,:])
    dists[i] = dist
    normals[i] = normal
    x = coordinates[i,0].numpy()
    nx = normal[0].numpy()
    y = coordinates[i,1].numpy()
    ny = normal[1].numpy()
    #plt.plot([coordinates[i,0], coordinates[i,0] + normal[0]],
    #        [coordinates[i,1], coordinates[i,1] + normal[1]], 'r-')


collision_coordinates = coordinates[dists < 0, :]
#plt.plot(collision_coordinates[:,0],
#        collision_coordinates[:,1], 'mo')

plt.scatter(coordinates[:,0].numpy(), coordinates[:,1].numpy(),
        c=dists.numpy(), cmap='viridis')


circle1 = plt.Circle((-0.4, 0.0), 0.15 + 0.04, color='b')
circle2 = plt.Circle((0.4, 0.0), 0.15 + 0.04, color='b')
circle3 = plt.Circle((-0.4, 0.0), 0.04, color='g')
circle4 = plt.Circle((0.4, 0.0), 0.04, color='g')
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.gca().add_patch(circle3)
plt.gca().add_patch(circle4)
plt.axis('equal')
plt.show()
"""

dynamics.render_traj(x_trj_initial, xg)
dynamics.render_traj_video(x_trj_initial, xg)

#plt.figure()
#plt.plot(x_trj_initial[:,0], x_trj_initial[:,1], 'r-')
#plt.show()

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
