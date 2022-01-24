import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from tqdm import tqdm

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 22})

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm
from alpha_gradient.lipschitz_estimator import estimate_lipschitz_probability

from ball_with_wall_dynamics import BallWithWallSoftDynamics
from ball_with_wall_dynamics_no_dome import BallWithWallSoftDynamicsNoDome

dynamics = BallWithWallSoftDynamics()
dynamics_no_dome = BallWithWallSoftDynamicsNoDome()
T = 83
dynamics.T = T
dynamics_no_dome.T = T

T_sub =70

batch_size = 15
arrow_scale = 0.01
distance_thrown = torch.zeros(batch_size)
distance_thrown_no_dome = torch.zeros(batch_size)
plt.figure()
plt.subplot(1,2,1)
theta_batch = torch.linspace(0.68, 0.73, batch_size)
for b in range(batch_size):
    x_trj = dynamics.rollout(theta_batch[b])
    plt.plot(x_trj[:,0], x_trj[:,1], color='magenta', alpha=0.5)

    plt.arrow(x_trj[-1,0], x_trj[-1,1], 
        arrow_scale * x_trj[-1,2], arrow_scale * x_trj[-1,3],
        color='magenta', alpha=0.5, head_width = 0.005,
        width=0.0005)
    plt.arrow(x_trj[T_sub,0], x_trj[T_sub,1], 
        arrow_scale * x_trj[T_sub,2], arrow_scale * x_trj[T_sub,3],
        color='magenta', alpha=0.5, head_width = 0.005,
        width=0.0005)        
    distance_thrown[b] = x_trj[dynamics.T, 0]
    
    x_trj = dynamics_no_dome.rollout(theta_batch[b])
    plt.plot(x_trj[:,0], x_trj[:,1], color='springgreen', alpha=0.5)
    plt.arrow(x_trj[-1,0], x_trj[-1,1], 
        arrow_scale * x_trj[-1,2], arrow_scale * x_trj[-1,3],
        color='springgreen', alpha=0.5, head_width = 0.005,
        width=0.0005)    
    plt.arrow(x_trj[T_sub,0], x_trj[T_sub,1], 
        arrow_scale * x_trj[T_sub,2], arrow_scale * x_trj[T_sub,3],
        color='springgreen', alpha=0.5, head_width = 0.005,
        width=0.0005)            
    distance_thrown_no_dome[b] = x_trj[dynamics.T, 0] 

dynamics.render(plt.gca())
plt.axis('equal')
plt.xlim([1.3, 1.6])
plt.ylim([0.4, 0.6])
plt.gca().axis('off')

plt.subplot(1,2,2)
batch_size = 100
distance_thrown = torch.zeros(batch_size)
distance_thrown_no_dome = torch.zeros(batch_size)
theta_batch = torch.linspace(0.5, 0.9, batch_size)
for b in range(batch_size):
    x_trj = dynamics.rollout(theta_batch[b])
    distance_thrown[b] = x_trj[dynamics.T, 0] 
    
    x_trj = dynamics_no_dome.rollout(theta_batch[b])
    distance_thrown_no_dome[b] = x_trj[dynamics.T, 0] 

plt.plot(
    theta_batch, -distance_thrown, '-',
    label='Smooth Geometry', color='magenta')
plt.plot(
    theta_batch, -distance_thrown_no_dome, '-',
    label='Non-Smooth Geometry', color='springgreen')
plt.xlabel('Angle Thrown')
plt.ylabel('Cost')
plt.legend()
plt.show()
