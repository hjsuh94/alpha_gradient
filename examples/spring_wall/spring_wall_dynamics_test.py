import numpy as np
import torch
import matplotlib.pyplot as plt 

from spring_wall_dynamics import SpringWallDynamics

dynamics = SpringWallDynamics(10000.0, 3.0)

T = 2000

x0 = torch.tensor([0.0, 100.0])
u_trj = 0.0 * torch.ones(T, 1)

x_trj = dynamics.rollout(x0, u_trj)

x0_batch = torch.zeros(1000, 2)
x0_batch[:,1] = 100.0
u_trj_batch = 30.0 *torch.rand(1000, T, 1)
x_trj_batch = dynamics.rollout_batch(x0_batch, u_trj_batch)

plt.figure()
plt.plot(x_trj_batch[0,:,0])
plt.plot(x_trj_batch[3,:,0])
plt.plot(x_trj_batch[5,:,0])
plt.show()
