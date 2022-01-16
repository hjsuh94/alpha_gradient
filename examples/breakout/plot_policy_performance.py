import numpy as np
import torch 
import matplotlib.pyplot as plt 

from alpha_gradient.objective_function_policy import ObjectiveFunctionPolicy
from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.optimizer_policy import (
    FobgdPolicyOptimizer, FobgdPolicyOptimizerParams,
    ZobgdPolicyOptimizer, ZobgdPolicyOptimizerParams,
    BCPolicyOptimizer, BCPolicyOptimizerParams)
from alpha_gradient.stepsize_scheduler import ManualScheduler   
from alpha_gradient.policy import LinearPolicy 
from alpha_gradient.plotting_tools import plot_cost

from breakout_dynamics_toi import BreakoutDynamics
from breakout_policyopt import BreakoutPolicyOpt

# Set up dynamics.
dynamics = BreakoutDynamics()
sample_size = 100
stdev = 0.001

# Initial condition.
xg = torch.tensor([0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
T = 200
Q = torch.diag(torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))
Qd = 100.0 * torch.diag(torch.tensor([1, 1, 0.1, 0.1, 0, 0, 0], dtype=torch.float32))
R = 0.001 * torch.diag(torch.tensor([1, 1, 1], dtype=torch.float32))

# Set up sampling function for x0.
def sample_x0_batch(sample_size):
    #ball_x0 = 1.8 * (2.0 * torch.rand(sample_size) - 1.0)
    #ball_y0 = 2.0 + torch.rand(sample_size)

    ball_x0 = 1.0 * torch.ones(sample_size) + torch.normal(
        0, 1.0, (sample_size,1)).squeeze(1)
    ball_y0 = 2.0 * torch.ones(sample_size) + torch.normal(
        0, 0.2, (sample_size,1)).squeeze(1)
    ball_vx0 = -0.2 * ball_x0 + torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)
    ball_vy0 = -0.2 * ball_y0 + torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)

    pad_x0 = torch.normal(0.0, 0.5, (sample_size,1)).squeeze(1)
    pad_y0 = torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)
    pad_theta0 = torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)

    return torch.vstack(
        (ball_x0, ball_y0, ball_vx0, ball_vy0, pad_x0, pad_y0, pad_theta0)
        ).transpose(0,1)

# Set up policy.
policy = LinearPolicy(dynamics.dim_x, dynamics.dim_u)
theta0 = torch.zeros(policy.d)

# Set up Objective.
objective = BreakoutPolicyOpt(T, dynamics, policy, Q, Qd, R, xg, sample_x0_batch)

#print(objective.zero_order_batch_gradient(theta0, sample_size, 0.01))
#print(objective.first_order_batch_gradient(theta0, sample_size, 0.01))

#============================================================================

# load policy parameters.
cost_fobg = np.load("fobg_cost.npy")
cost_zobg = np.load("zobg_cost.npy")

# Data cleaning.

policy_fobg = np.load("fobg_params.npy", allow_pickle=True)
policy_zobg = np.load("zobg_params.npy", allow_pickle=True)

theta_fobg = policy_fobg[1000]
theta_zobg = policy_zobg[1000]

plt.figure()
plot_cost(plt.gca(), cost_fobg, 30, 'red', label='fobg')
plot_cost(plt.gca(), cost_zobg, 30, 'blue', label='zobg')
plt.legend()
plt.show()

plt.figure(figsize=(16,12))
plt.subplot(1,2,1)
x_trj_batch, _ = objective.rollout_policy_batch(
    sample_x0_batch(500), torch.zeros(500, T, objective.m),
    torch.tensor(theta_fobg))
for b in range(x_trj_batch.shape[0]):
    plt.plot(x_trj_batch[b,:,0], x_trj_batch[b,:,1], 'k-', alpha=0.05)
plt.plot(x_trj_batch[:,0,0], x_trj_batch[:,0,1], 'ko', alpha=0.6,
    markersize=2.0, label='initial')
plt.plot(x_trj_batch[:,-1,0], x_trj_batch[:,-1,1], 'ro', alpha=0.6,
    markersize=2.0, label='final')
plt.plot(0.0, 2.5, 'o', color='springgreen', alpha=0.6,
    markersize=10.0, label='target point')    
polygon = np.array([
    dynamics.x_width * np.array([1, 1, -1, -1]),
    dynamics.y_width * np.array([1, -1, -1, 1]),
])
plt_polygon = plt.Polygon(
    np.transpose(polygon), facecolor='springgreen',
    edgecolor='springgreen', alpha=0.1)
plt.gca().add_patch(plt_polygon)    
plt.legend()

plt.xlim([-dynamics.x_width, dynamics.x_width])
plt.ylim([-dynamics.y_width, dynamics.y_width])

plt.subplot(1,2,2)
x_trj_batch, _ = objective.rollout_policy_batch(
    sample_x0_batch(500), torch.zeros(500, T, objective.m),
    torch.tensor(theta_zobg))
for b in range(x_trj_batch.shape[0]):
    plt.plot(x_trj_batch[b,:,0], x_trj_batch[b,:,1], 'k-', alpha=0.05)
plt.plot(x_trj_batch[:,0,0], x_trj_batch[:,0,1], 'ko', alpha=0.6,
    markersize=2.0, label='initial')
plt.plot(x_trj_batch[:,-1,0], x_trj_batch[:,-1,1], 'bo', alpha=0.6,
    color='blue', markersize=2.0, label='final')
plt.plot(0.0, 2.5, 'o', color='springgreen', alpha=0.6,
    markersize=10.0, label='target point')
plt_polygon = plt.Polygon(
    np.transpose(polygon), facecolor='springgreen',
    edgecolor='springgreen', alpha=0.1)
plt.gca().add_patch(plt_polygon)        
plt.legend()

plt.xlim([-dynamics.x_width, dynamics.x_width])
plt.ylim([-dynamics.y_width, dynamics.y_width])

dynamics.render_traj_batch(x_trj_batch[0:20], np.array([0.0, 2.5]))

plt.show()


