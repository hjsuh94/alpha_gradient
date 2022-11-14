import numpy as np
import torch 
import matplotlib.pyplot as plt 
from tqdm import tqdm

from alpha_gradient.objective_function_policy import ObjectiveFunctionPolicy
from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.optimizer_policy import (
    FobgdPolicyOptimizer, FobgdPolicyOptimizerParams,
    ZobgdPolicyOptimizer, ZobgdPolicyOptimizerParams,
    BCPolicyOptimizer, BCPolicyOptimizerParams)
from alpha_gradient.stepsize_scheduler import ManualScheduler   
from alpha_gradient.policy import LinearPolicy 

#from breakout_dynamics_smooth_geom import BreakoutDynamics
from breakout_dynamics_implicit_geom import BreakoutDynamics
from breakout_policyopt import BreakoutPolicyOpt
from initial_distribution import (
    sample_x0_batch, sample_x0_batch_narrow,
    sample_x0_batch_single, sample_x0_batch_vert)

# Set up dynamics.

sample_size = 1000
stdev = 0.01

#kappa_array = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4])
#kappa_array = [10]

kappa_array = np.linspace(1.99, 2.0, 11)
# Initial condition.
xg = torch.tensor([0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
T = 200
Q = torch.diag(torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))
Qd = 100.0 * torch.diag(torch.tensor([1, 1, 0.1, 0.1, 0, 0, 0], dtype=torch.float32))
R = 0.1 * torch.diag(torch.tensor([1, 1, 1], dtype=torch.float32))

for i in tqdm(range(len(kappa_array))):

    kappa = np.power(10.0, kappa_array[i])
    print(kappa_array[i])
    dynamics = BreakoutDynamics()
    dynamics.kappa = kappa

    # Set up policy.
    policy = LinearPolicy(dynamics.dim_x, dynamics.dim_u)
    theta0 = 0.0 * torch.ones(policy.d)

    # Set up Objective.
    objective = BreakoutPolicyOpt(T, dynamics, policy, Q, Qd, R, xg,
        sample_x0_batch)

    #print(objective.zero_order_batch_gradient(theta0, sample_size, 0.0001))
    #print(objective.first_order_batch_gradient(theta0, sample_size, 0.0001))

    #============================================================================
    print("FOBG Optimization, Kappa={:.3f}".format(kappa_array[i]))
    params = FobgdPolicyOptimizerParams()
    params.stdev = stdev
    params.sample_size = sample_size
    def constant_step(iter, initial_step): return 1e-6 * 1/(iter ** 0.1)
    params.step_size_scheduler = ManualScheduler(constant_step, 1e-6)
    params.theta0 = theta0
    params.filename = "fobg_softplus_{:.3f}".format(kappa_array[i])
    #params.filename = "fobg_relu_highvar"
    num_iters = 200

    optimizer = FobgdPolicyOptimizer(objective, params)
    optimizer.iterate(num_iters)

    x_trj_batch, _ = objective.rollout_policy_batch(
        sample_x0_batch(100), torch.zeros(100, T, objective.m),
        optimizer.theta)

    #dynamics.render_traj_batch(x_trj_batch, xg)

    #plt.figure()
    #for b in range(x_trj_batch.shape[0]):
    #    plt.plot(x_trj_batch[b,:,0], x_trj_batch[b,:,1])
    #plt.show()

    """
    #============================================================================
    print("ZOBG Optimization, Kappa={:.1f}".format(kappa_array[i]))    
    params = ZobgdPolicyOptimizerParams()
    params.stdev = stdev
    params.sample_size = sample_size
    def constant_step(iter, initial_step): return 1e-6 * 1/(iter ** 0.1)
    params.step_size_scheduler = ManualScheduler(constant_step, 1e-6)
    params.theta0 = theta0
    params.filename = "zobg_softplus_{:.1f}".format(kappa_array[i])  
    #params.filename = "zobg_relu_highvar"
    num_iters = 200


    optimizer = ZobgdPolicyOptimizer(objective, params)
    optimizer.iterate(num_iters)
    
    x_trj_batch, _ = objective.rollout_policy_batch(
        sample_x0_batch(100), torch.zeros(100, T, objective.m),
        optimizer.theta)
    """

plt.figure()
for b in range(x_trj_batch.shape[0]):
    plt.plot(x_trj_batch[b,:,0], x_trj_batch[b,:,1])
plt.show()

plt.figure()
plt.plot(optimizer.cost_lst)
plt.show()


