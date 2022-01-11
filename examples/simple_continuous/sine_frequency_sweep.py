import numpy as np
import time
import warnings
warnings.filterwarnings('error')
import matplotlib.pyplot as plt

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm
from sine_wave import Sine

from tqdm import tqdm

dmax = 100
omega_space = np.linspace(1,10,dmax)
n_gradient_samples = 100
n_samples = 100
sigma = 1.0

fov_storage = np.zeros(dmax)
zov_storage = np.zeros(dmax)

for i in tqdm(range(len(omega_space))):
    func = Sine(omega_space[i])

    x = np.array([0.0])

    fobg_storage = np.zeros((n_gradient_samples,1))
    zobg_storage = np.zeros((n_gradient_samples,1))    

    for k in range(n_gradient_samples):
        fobg_storage[k,:] = func.first_order_batch_gradient(
            x, n_samples, sigma)
        zobg_storage[k,:] = func.zero_order_batch_gradient(
            x, n_samples, sigma)

    fov_storage[i] = compute_variance_norm(fobg_storage, 'fro')
    zov_storage[i] = compute_variance_norm(zobg_storage, 'fro')

plt.figure()
plt.plot(omega_space, fov_storage, 'r-', label='FOBG Variance')
plt.plot(omega_space, zov_storage, 'b-', label='ZOBG Variance')
plt.legend()
plt.xlabel('frequency (Hz)')
plt.ylabel('Variance')
plt.show()