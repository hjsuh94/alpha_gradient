import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from pydrake.all import InitializeAutoDiff, ExtractGradient
from jump_sgd.objective_function import ObjectiveFunction
from jump_sgd.statistical_analysis import compute_mean, compute_variance
from norms import LpNorm

p = 5
dmax = 50
n_gradient_samples = 100
n_samples = 1000
sigma = 0.2

fom_storage = []
zom_storage = []
fom_zom_diff = np.zeros(dmax)

fov_storage = np.zeros(dmax)
zov_storage = np.zeros(dmax)

fov_storage_nuc = np.zeros(dmax)
zov_storage_nuc = np.zeros(dmax)

fov_storage_fro = np.zeros(dmax)
zov_storage_fro = np.zeros(dmax)

for d in tqdm(range(1,dmax+1)):
    lp_norm = LpNorm(p, d)
    x = np.ones(d)

    fobg_storage = np.zeros((n_gradient_samples, d))
    zobg_storage = np.zeros((n_gradient_samples, d))

    for k in range(n_gradient_samples):
        fobg_storage[k,:] = lp_norm.first_order_batch_gradient(
            x, n_samples, sigma)
        zobg_storage[k,:] = lp_norm.zero_order_batch_gradient(
            x, n_samples, sigma)

    fom = compute_mean(fobg_storage)
    zom = compute_mean(zobg_storage)

    fom_zom_diff[d-1] = np.linalg.norm(fom - zom, 2)
    fov_storage[d-1] = compute_variance(fobg_storage)
    zov_storage[d-1] = compute_variance(zobg_storage)

    fov_storage_nuc[d-1] = compute_variance(fobg_storage, 'nuc')
    zov_storage_nuc[d-1] = compute_variance(zobg_storage, 'nuc')

    fov_storage_fro[d-1] = compute_variance(fobg_storage, 'fro')
    zov_storage_fro[d-1] = compute_variance(zobg_storage, 'fro')        

plt.figure()
plt.plot(range(d), fov_storage, 'r-', label='Variance of FOBG (2-norm)')
plt.plot(range(d), zov_storage, 'b-', label='Variance of ZOBG (2-norm)')

plt.plot(range(d), fov_storage_nuc, 'r--', label='Variance of FOBG (nuc-norm)')
plt.plot(range(d), zov_storage_nuc, 'b--', label='Variance of ZOBG (nuc-norm)')

plt.plot(range(d), fov_storage_fro, 'r-*', label='Variance of FOBG (fro-norm)')
plt.plot(range(d), zov_storage_fro, 'b-*', label='Variance of ZOBG (fro-norm)')

plt.plot(range(d), fom_zom_diff, color='springgreen',
    label='Diff between FOBM and ZOBM (2-norm)')
plt.xlabel('Dimension (d)')
plt.legend()
plt.show()
