import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm
from ball_with_wall import BallWithWall
from examples.ball_with_wall_soft.ball_with_wall_objective import BallWithWallTorch

dmax = 100
n_gradient_samples = 100
n_samples = 100
sigma = 0.05
coordinate_range = np.linspace(0, np.pi/2, dmax)
alpha = 0.5

eval_storage = np.zeros(dmax)
fom_storage = np.zeros(dmax)
zom_storage = np.zeros(dmax)
aom_storage = np.zeros(dmax)
fom_zom_diff = np.zeros(dmax)
aom_zom_diff = np.zeros(dmax)

fov_storage = np.zeros(dmax)
zov_storage = np.zeros(dmax)
aov_storage = np.zeros(dmax)

for i in tqdm(range(len(coordinate_range))):
    d = 1
    lp_norm = BallWithWallTorch()

    xval = coordinate_range[i]
    x = xval * np.ones(d)

    eval_storage[i] = lp_norm.evaluate(x, np.array([0.0]))

    fobg_storage = np.zeros((n_gradient_samples, d))
    zobg_storage = np.zeros((n_gradient_samples, d))
    aobg_storage = np.zeros((n_gradient_samples, d))

    for k in range(n_gradient_samples):
        fobg_storage[k,:] = lp_norm.first_order_batch_gradient(
            x, n_samples, sigma/d)
        zobg_storage[k,:] = lp_norm.zero_order_batch_gradient(
            x, n_samples, sigma/d)
        aobg_storage[k,:] = alpha * fobg_storage[k,:] + (1-alpha) * zobg_storage[k,:]

    fom = compute_mean(fobg_storage)
    zom = compute_mean(zobg_storage)
    aom = compute_mean(aobg_storage)    

    fom_zom_diff[i] = np.linalg.norm(fom - zom)
    aom_zom_diff[i] = np.linalg.norm(aom - zom)    
    fom_storage[i] = np.linalg.norm(fom)
    zom_storage[i] = np.linalg.norm(zom)
    aom_storage[i] = np.linalg.norm(aom)    

    fov_storage[i] = compute_variance_norm(fobg_storage, 'fro')
    zov_storage[i] = compute_variance_norm(zobg_storage, 'fro')
    aov_storage[i] = compute_variance_norm(aobg_storage, 'fro')

plt.figure(figsize=(8,4))
plt.subplot(3,1,1)
plt.plot(coordinate_range, fov_storage, 'r-', label=('FOBG Variance'))
plt.plot(coordinate_range, zov_storage, color='royalblue', label=('ZOBG Variance'))
plt.plot(coordinate_range, aov_storage, color='springgreen', label=('AOBG Variance'))
plt.xlabel('Angle thrown (theta)')
plt.legend()

plt.subplot(3,1,2)
plt.plot(coordinate_range, fom_zom_diff, 'r-', label=('FOBG Bias'))
plt.plot(coordinate_range, np.zeros(dmax), color='royalblue', label=('ZOBG Bias'))
plt.plot(coordinate_range, aom_zom_diff, color='springgreen', label=('AOBG Bias'))
plt.xlabel('Angle thrown (theta)')
plt.legend()

plt.subplot(3,1,3)
plt.plot(coordinate_range, eval_storage, 'k-', label=('Cost Landscape'))
plt.xlabel('Angle thrown (theta)')
plt.legend()
plt.savefig("results_zero.png")
plt.show()
