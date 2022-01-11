import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm
from soft_heaviside import SoftHeaviside

dmax = 50
n_gradient_samples = 100
n_samples = 100
sigma = 0.1
coordinate_range = np.linspace(-1, 1, dmax)
alpha = 0.5

v_storage = np.zeros(dmax)
fov_storage = np.zeros(dmax)
zov_storage = np.zeros(dmax)

for i in tqdm(range(len(coordinate_range))):
    d = 1
    lp_norm = SoftHeaviside(0.4)

    xval = coordinate_range[i]
    x = xval * np.ones(d)

    v_storage[i] = lp_norm.evaluate(x, np.array([0.0]))

    fobg_storage = np.zeros((n_gradient_samples, d))
    zobg_storage = np.zeros((n_gradient_samples, d))
    aobg_storage = np.zeros((n_gradient_samples, d))

    for k in range(n_gradient_samples):
        fobg_storage[k,:] = lp_norm.first_order_batch_gradient(
            x, n_samples, sigma/d)
        zobg_storage[k,:] = lp_norm.zero_order_batch_gradient(
            x, n_samples, sigma/d)
        aobg_storage[k,:] = alpha * fobg_storage[k,:] + (1-alpha) * zobg_storage[k,:]

    fov_storage[i] = compute_variance_norm(fobg_storage, 'fro')
    zov_storage[i] = compute_variance_norm(zobg_storage, 'fro')

plt.figure()
plt.subplot(2,3,1)
plt.plot(coordinate_range, fov_storage, 'r-', label=('FOBG Var'))
plt.plot(coordinate_range, zov_storage, color='royalblue', label=('ZOBG Var'))
plt.xlabel('X Coordinate')
plt.title('delta = 0.5')
plt.legend()

plt.subplot(2,3,4)
plt.plot(coordinate_range, v_storage, 'k-', label='Objective')
plt.xlabel('X coordinate')
plt.legend()


v_storage = np.zeros(dmax)
fov_storage = np.zeros(dmax)
zov_storage = np.zeros(dmax)

for i in tqdm(range(len(coordinate_range))):
    d = 1
    lp_norm = SoftHeaviside(0.1)

    xval = coordinate_range[i]
    x = xval * np.ones(d)

    v_storage[i] = lp_norm.evaluate(x, np.array([0.0]))

    fobg_storage = np.zeros((n_gradient_samples, d))
    zobg_storage = np.zeros((n_gradient_samples, d))
    aobg_storage = np.zeros((n_gradient_samples, d))

    for k in range(n_gradient_samples):
        fobg_storage[k,:] = lp_norm.first_order_batch_gradient(
            x, n_samples, sigma/d)
        zobg_storage[k,:] = lp_norm.zero_order_batch_gradient(
            x, n_samples, sigma/d)
        aobg_storage[k,:] = alpha * fobg_storage[k,:] + (1-alpha) * zobg_storage[k,:]

    fov_storage[i] = compute_variance_norm(fobg_storage, 'fro')
    zov_storage[i] = compute_variance_norm(zobg_storage, 'fro')

plt.subplot(2,3,2)
plt.plot(coordinate_range, fov_storage, 'r-', label=('FOBG Var'))
plt.plot(coordinate_range, zov_storage, color='royalblue', label=('ZOBG Var'))
plt.xlabel('X Coordinate')
plt.title('delta = 0.1')
plt.legend()

plt.subplot(2,3,5)
plt.plot(coordinate_range, v_storage, 'k-', label='Objective')
plt.xlabel('X coordinate')
plt.legend()



v_storage = np.zeros(dmax)
fov_storage = np.zeros(dmax)
zov_storage = np.zeros(dmax)

for i in tqdm(range(len(coordinate_range))):
    d = 1
    lp_norm = SoftHeaviside(0.01)

    xval = coordinate_range[i]
    x = xval * np.ones(d)

    v_storage[i] = lp_norm.evaluate(x, np.array([0.0]))

    fobg_storage = np.zeros((n_gradient_samples, d))
    zobg_storage = np.zeros((n_gradient_samples, d))
    aobg_storage = np.zeros((n_gradient_samples, d))

    for k in range(n_gradient_samples):
        fobg_storage[k,:] = lp_norm.first_order_batch_gradient(
            x, n_samples, sigma/d)
        zobg_storage[k,:] = lp_norm.zero_order_batch_gradient(
            x, n_samples, sigma/d)
        aobg_storage[k,:] = alpha * fobg_storage[k,:] + (1-alpha) * zobg_storage[k,:]

    fov_storage[i] = compute_variance_norm(fobg_storage, 'fro')
    zov_storage[i] = compute_variance_norm(zobg_storage, 'fro')

plt.subplot(2,3,3)
plt.plot(coordinate_range, fov_storage, 'r-', label=('FOBG Var'))
plt.plot(coordinate_range, zov_storage, color='royalblue', label=('ZOBG Var'))
plt.xlabel('X Coordinate')
plt.title('delta = 0.01')
plt.legend()

plt.subplot(2,3,6)
plt.plot(coordinate_range, v_storage, 'k-', label='Objective')
plt.xlabel('X coordinate')
plt.legend()





plt.show()