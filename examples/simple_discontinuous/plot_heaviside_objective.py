import numpy as np

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction
import matplotlib.pyplot as plt
from heaviside_objective import HeavisideAllPositive

heaviside = HeavisideAllPositive(2)

x_space = np.linspace(-1, 1, 100)
y_space = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_space,y_space)

positions = np.vstack([X.ravel(), Y.ravel()]).transpose()
eval = np.zeros(positions.shape[0])
eval_smooth = np.zeros(positions.shape[0])
samples = np.random.normal(0, 0.2, size=(1000, 2))
for k in range(positions.shape[0]):
    eval[k] = heaviside.evaluate(positions[k], np.zeros(2))
    eval_smooth[k] = np.average(
        heaviside.evaluate_batch(positions[k], samples), axis=0)

Z = eval.reshape(100, 100)
Z_smooth = eval_smooth.reshape(100, 100)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(Y, X, Z, cmap='plasma', alpha=0.5, label='heaviside')
ax.plot_surface(Y, X, Z_smooth, cmap='viridis', alpha=0.8, label='stochastic')

plt.xlabel('x')
plt.ylabel('y')
plt.show()


