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
from alpha_gradient.lipschitz_estimator import estimate_lipschitz_probability
from pivot_torch import PivotTorch

dmax = 100
n_gradient_samples = 100
n_samples = 1000
sigma = 0.02
coordinate_range = np.linspace(0, np.pi/2, dmax)

obj_storage = np.zeros(dmax)
bobj_storage = np.zeros(dmax)
fom_storage = np.zeros(dmax)
zom_storage = np.zeros(dmax)
aom_storage = np.zeros(dmax)
fom_zom_diff = np.zeros(dmax)
aom_zom_diff = np.zeros(dmax)

fov_storage = np.zeros(dmax)
zov_storage = np.zeros(dmax)
aov_storage = np.zeros(dmax)

alpha_storage = np.zeros(dmax)

lp_norm = PivotTorch()

for i in tqdm(range(len(coordinate_range))):
    d = 1

    xval = coordinate_range[i]
    x = xval * np.ones(d)

    obj_storage[i] = lp_norm.evaluate(x, np.array([0.0]))
    bobj_storage[i],_ = lp_norm.bundled_objective(x, n_samples, sigma)
    fobg_storage = np.zeros((n_gradient_samples, d))
    zobg_storage = np.zeros((n_gradient_samples, d))
    aobg_storage = np.zeros((n_gradient_samples, d))
    alpha_substorage = np.zeros(n_gradient_samples)

    for k in range(n_gradient_samples):
        fobg_storage[k,:], _ = lp_norm.first_order_batch_gradient(
            x, n_samples, sigma/d)
        zobg_storage[k,:], _ = lp_norm.zero_order_batch_gradient(
            x, n_samples, sigma/d)
        aobg_storage[k,:], alpha_substorage[k] = lp_norm.bias_constrained_aobg(
            x, n_samples, sigma/d, 0.1, L=1.0)

    fom = compute_mean(fobg_storage)
    zom = compute_mean(zobg_storage)
    aom = compute_mean(aobg_storage)

    alpha_storage[i] = compute_mean(alpha_substorage)

    fom_zom_diff[i] = np.linalg.norm(fom - zom)
    aom_zom_diff[i] = np.linalg.norm(aom - zom)    
    fom_storage[i] = np.linalg.norm(fom)
    zom_storage[i] = np.linalg.norm(zom)
    aom_storage[i] = np.linalg.norm(aom)    

    fov_storage[i] = np.sqrt(compute_variance_norm(fobg_storage, 'fro'))
    zov_storage[i] = np.sqrt(compute_variance_norm(zobg_storage, 'fro'))
    aov_storage[i] = np.sqrt(compute_variance_norm(aobg_storage, 'fro'))

plt.figure(figsize=(8,4))
plt.subplot(4,1,1)
plt.plot(coordinate_range, fov_storage, 'r-', label=('FOBG Variance'))
plt.plot(coordinate_range, zov_storage, color='royalblue', label=('ZOBG Variance'))
plt.plot(coordinate_range, aov_storage, color='springgreen', label=('AOBG Variance'))
plt.xlabel('Angle thrown (theta)')
plt.legend()

plt.subplot(4,1,2)
plt.plot(coordinate_range, fom_zom_diff, 'r-', label=('FOBG Bias'))
plt.plot(coordinate_range, np.zeros(dmax), color='royalblue', label=('ZOBG Bias'))
plt.plot(coordinate_range, aom_zom_diff, color='springgreen', label=('AOBG Bias'))
plt.xlabel('Angle thrown (theta)')
plt.legend()

plt.subplot(4,1,3)
plt.plot(coordinate_range, obj_storage, 'k-', label=(r'$f(x,0)$'))
plt.plot(coordinate_range, bobj_storage, 'm-', label=(r'$F(x)$'))
plt.plot(0.7394,
    lp_norm.evaluate(0.7394 * np.ones(1), np.zeros(1)), '*',
    color='royalblue', markersize=20, alpha=0.6)
plt.plot(0.8434,
    lp_norm.evaluate(0.8434 * np.ones(1), np.zeros(1)), '*',
    color='red', markersize=20, alpha=0.6)
plt.plot(0.7577,
    lp_norm.evaluate(0.7577 * np.ones(1), np.zeros(1)), '*',
    color='springgreen', markersize=20, alpha=0.6)
plt.plot(0.1,
    lp_norm.evaluate(0.1 * np.ones(1), np.zeros(1)), '^',
    color='black', markersize=20, alpha=0.6)    
plt.xlabel('Angle thrown (theta)')
plt.legend()

plt.subplot(4,1,4)
plt.plot(coordinate_range, alpha_storage, color='purple', label=('alpha'))
plt.xlabel('Angle thrown (theta)')
plt.legend()
plt.show()
