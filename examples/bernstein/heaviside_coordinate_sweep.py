import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction
from heaviside_objective import HeavisideAllPositive
from alpha_gradient.lipschitz_estimator import (
    compute_pairwise_lipschitz_matrix, compute_pairwise_lipschitz_vector,
    compute_pairwise_lipschitz_tensor, estimate_lipschitz_probability)
from alpha_gradient.statistical_analysis import compute_confidence_interval

import warnings 
warnings.filterwarnings("error")

heaviside = HeavisideAllPositive(1)

sweep = 100
xspace = np.linspace(-1, 1, sweep)
eps_lst_tight = np.zeros((sweep,2))
eps_lst_loose = np.zeros((sweep,2))
value_lst = np.zeros(sweep)
n_samples = 1000
sigma = 0.2

for i in range(sweep):
    x = xspace[i] * np.ones(1)
    fobg_mu, fobg_sigma = heaviside.first_order_batch_gradient(x, n_samples, sigma)
    zobg_mu, zobg_sigma = heaviside.zero_order_batch_gradient(x, n_samples, sigma)

    value_lst[i] = zobg_mu

    ci_tight = compute_confidence_interval(
        zobg_mu, zobg_sigma ** 2.0, 1000, 10.0, 0.95)
    ci_loose = compute_confidence_interval(
        zobg_mu, zobg_sigma ** 2.0, 100, 10.0, 0.95)        

    eps_lst_tight[i,0] = ci_tight[0]
    eps_lst_tight[i,1] = ci_tight[1]    
    eps_lst_loose[i,0] = ci_loose[0]
    eps_lst_loose[i,1] = ci_loose[1]    

plt.figure()
plt.plot(xspace, value_lst, 'k-')
plt.gca().fill_between(
    xspace, value_lst + eps_lst_tight[:,0],
    value_lst + eps_lst_tight[:,1], alpha=0.2,
    color='red', label='1000 Samples')
plt.gca().fill_between(
    xspace, value_lst + eps_lst_loose[:,0],
    value_lst + eps_lst_loose[:,1], alpha=0.2,
    color='blue', label='100 Samples')    
plt.legend()
plt.show()

    
    
    