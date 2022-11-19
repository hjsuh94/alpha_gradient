import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import torch

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm
from ball_with_wall_softplus import BallWithWallSoftPlus
from ball_with_wall import BallWithWallSoftPlusObjective


plt.figure()
colors = 'rgby'
B = 10
plot_space = np.arange(0, B, 40)

dynamics = BallWithWallSoftPlus()
dynamics.T = 250
dynamics.kappa = np.power(10.0, 1)
objective = BallWithWallSoftPlusObjective(dynamics)
theta_linspace = 0.5661 * torch.ones(B)
#theta_linspace = 1.0 * torch.ones(B)
objects = objective.evaluate_batch(theta_linspace, torch.zeros(B))
dynamics.render(plt.gca())
trj_batch = dynamics.rollout_batch(theta_linspace)
plt.plot(trj_batch[0,:,0], trj_batch[0,:,1], 'r')
#print(trj_batch[0,:,:].shape)
dists, normals = dynamics.compute_sg_batch(trj_batch[0,:,0:2])
print(trj_batch[0,:,0:2].shape)
print(dists.shape)
print(normals.shape)
plt.quiver(trj_batch[0,:,0], trj_batch[0,:,1], normals[:,0], normals[:,1])
plt.axis('equal')
plt.show()
