import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.optimizer import (
    FobgdOptimizer, FobgdOptimizerParams,
    ZobgdOptimizer, ZobgdOptimizerParams,
    AobgdOptimizer, AobgdOptimizerParams)
from alpha_gradient.stepsize_scheduler import ManualScheduler
from heaviside_objective import HeavisideAllPositive

plt.figure()
rect = patches.Rectangle(
    (0, 0), 0.3, 0.3, linewidth=1, edgecolor=(0,0,0,1),
    facecolor=(1,0,0,0.5))
plt.gca().add_patch(rect)
rect = patches.Rectangle(
    (0, 0), -0.3, 0.3, linewidth=1, edgecolor=(0,0,0,0),
    facecolor=(0,0,1,0.5))
plt.gca().add_patch(rect)
rect = patches.Rectangle(
    (0, 0), -0.3, -0.3, linewidth=1, edgecolor=(0,0,0,0),
    facecolor=(0,0,1,0.5))
plt.gca().add_patch(rect)
rect = patches.Rectangle(
    (0, 0), 0.3, -0.3, linewidth=1, edgecolor=(0,0,0,0),
    facecolor=(0,0,1,0.5))
plt.gca().add_patch(rect)


params = ZobgdOptimizerParams
params.variance = 2.0
params.sample_size = 100

initial_step = 1e-1
def step_size_schedule(iter, initial_step):
    return initial_step
params.step_size_scheduler = ManualScheduler(step_size_schedule, initial_step)

params.x0_initial = 0.1 * np.array([1, 1])
objective = HeavisideAllPositive(2)

optimizer = ZobgdOptimizer(objective, params)
optimizer.iterate(30)

descent_array = np.array(optimizer.x_lst)

plt.plot(descent_array[:,0], descent_array[:,1], 'o-', color='royalblue',
    label='ZOBG')

#====================================================

params = AobgdOptimizerParams
params.variance = 2.0
params.sample_size = 100
params.alpha = 0.5

initial_step = 1e-1
def step_size_schedule(iter, initial_step):
    return initial_step
params.step_size_scheduler = ManualScheduler(step_size_schedule, initial_step)

params.x0_initial = 0.1 * np.array([1, 1])
objective = HeavisideAllPositive(2)

optimizer = AobgdOptimizer(objective, params)
optimizer.iterate(30)

descent_array = np.array(optimizer.x_lst)
plt.plot(descent_array[:,0], descent_array[:,1], 'o-', color='springgreen',
    label='AOBG')

#====================================================

params = FobgdOptimizerParams
params.variance = 2.0
params.sample_size = 100

initial_step = 1e-1
def step_size_schedule(iter, initial_step):
    return initial_step
params.step_size_scheduler = ManualScheduler(step_size_schedule, initial_step)

params.x0_initial = 0.1 * np.array([1, 1])
objective = HeavisideAllPositive(2)

optimizer = FobgdOptimizer(objective, params)
optimizer.iterate(30)

descent_array = np.array(optimizer.x_lst)
plt.plot(descent_array[:,0], descent_array[:,1], 'ro-',
    label='FOBG')
plt.legend()
plt.show()
