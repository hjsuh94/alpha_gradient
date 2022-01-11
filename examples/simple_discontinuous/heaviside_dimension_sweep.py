import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm
from heaviside_objective import HeavisideAllPositive

dmax = 50
n_gradient_samples = 1000
n_samples = 100
sigma = 2.0

fom_storage = np.zeros(dmax)
zom_storage = np.zeros(dmax)
fom_zom_diff = np.zeros(dmax)

fov_storage = np.zeros(dmax)
zov_storage = np.zeros(dmax)
for d in tqdm(range(1,dmax+1)):
    lp_norm = HeavisideAllPositive(d)

    x = np.zeros(d)

    fobg_storage = np.zeros((n_gradient_samples, d))
    zobg_storage = np.zeros((n_gradient_samples, d))

    for k in range(n_gradient_samples):
        fobg_storage[k,:] = lp_norm.first_order_batch_gradient(
            x, n_samples, sigma/np.sqrt(d))
        zobg_storage[k,:] = lp_norm.zero_order_batch_gradient(
            x, n_samples, sigma/np.sqrt(d))

    fom = compute_mean(fobg_storage)
    zom = compute_mean(zobg_storage)

    fom_zom_diff[d-1] = np.linalg.norm(zom - fom)
    fom_storage[d-1] = np.linalg.norm(fom)
    zom_storage[d-1] = np.linalg.norm(zom)

    fom_zom_diff[d-1] = np.linalg.norm(fom - zom, 2)
    fov_storage[d-1] = compute_variance_norm(fobg_storage, 'fro')
    zov_storage[d-1] = compute_variance_norm(zobg_storage, 'fro')

plt.figure(figsize=(8,4))
plt.subplot(2,1,1)
plt.plot(range(d), fov_storage, 'r-', label=('FOBG Variance'))
plt.plot(range(d), zov_storage, 'b-', label=('ZOBG Variance'))
plt.xlabel('Dimension (d)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(range(d), fom_zom_diff, 'r-', label='FOBG Bias (Diff with ZOBG)')
plt.plot(range(d), np.zeros(d), 'b-', label='ZOBG Bias')
plt.xlabel('Dimension (d)')
plt.legend()
#plt.savefig("results_zero.png")
plt.show()
