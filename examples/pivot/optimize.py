import numpy as np
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
from alpha_gradient.optimizer import (
    FobgdOptimizer, FobgdOptimizerParams,
    ZobgdOptimizer, ZobgdOptimizerParams,
    BiasConstrainedOptimizer, BiasConstrainedOptimizerParams)
from alpha_gradient.stepsize_scheduler import ManualScheduler

from pivot_torch import PivotTorch

num_trials = 20
num_iters = 500
num_samples = 50

def constant_step(iter, initial_step):
    return 0.01

objective = PivotTorch()
#=============================================================================

params = ZobgdOptimizerParams()
params.stdev = 0.02
params.sample_size = num_samples
params.verbose = False
params.step_size_scheduler = ManualScheduler(constant_step, 0.1)
params.x0_initial = np.array([0.1])

zobgd_storage = np.zeros((num_trials, num_iters + 2))

for i in tqdm(range(num_trials)):
    optimizer = ZobgdOptimizer(objective, params)
    optimizer.iterate(num_iters)
    zobgd_storage[i,:] = optimizer.cost_lst

zobgd_mean = np.mean(zobgd_storage, axis=0)
zobgd_std = np.std(zobgd_storage, axis=0)

print(optimizer.x)

#=============================================================================

params = FobgdOptimizerParams()
params.stdev = 0.02
params.sample_size = num_samples
params.verbose = False
params.step_size_scheduler = ManualScheduler(constant_step, 0.1)
params.x0_initial = np.array([0.1])

fobgd_storage = np.zeros((num_trials, num_iters + 2))

for i in tqdm(range(num_trials)):
    optimizer = FobgdOptimizer(objective, params)
    optimizer.iterate(num_iters)
    fobgd_storage[i,:] = optimizer.cost_lst

fobgd_mean = np.mean(fobgd_storage, axis=0)
fobgd_std = np.std(fobgd_storage, axis=0)

print(optimizer.x)

#=============================================================================

params = BiasConstrainedOptimizerParams()
params.stdev = 0.02
params.sample_size = num_samples
params.verbose = False
params.step_size_scheduler = ManualScheduler(constant_step, 0.1)
params.x0_initial = np.array([0.1])

params.delta = 0.95
params.L = 1.0
params.gamma = 0.2


aobgd_storage = np.zeros((num_trials, num_iters + 2))

for i in tqdm(range(num_trials)):
    optimizer = BiasConstrainedOptimizer(objective, params)
    optimizer.iterate(num_iters)
    aobgd_storage[i,:] = optimizer.cost_lst

aobgd_mean = np.mean(aobgd_storage, axis=0)
aobgd_std = np.std(aobgd_storage, axis=0)

print(optimizer.x)

#=============================================================================

plt.figure()

scale = 10

plt.fill_between(range(num_iters + 2),
    zobgd_mean - scale * zobgd_std, zobgd_mean + scale * zobgd_std,
    color='royalblue', alpha=0.2)
plt.plot(zobgd_mean, color='royalblue', label='ZOBG')

plt.fill_between(range(num_iters + 2),
    aobgd_mean - scale * aobgd_std, aobgd_mean + scale * aobgd_std,
    color='springgreen', alpha=0.2)
plt.plot(aobgd_mean, color='springgreen', label='AOBG')

plt.fill_between(range(num_iters + 2),
    fobgd_mean - scale * fobgd_std, fobgd_mean + scale * fobgd_std,
    color='red', alpha=0.2)
plt.plot(fobgd_mean, color='red', label='FOBG')

plt.xlabel('iterations')
plt.ylabel('cost')
plt.legend()

plt.show()
