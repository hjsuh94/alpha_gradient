import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction
from ball_with_wall import BallWithWall
from alpha_gradient.lipschitz_estimator import (
    compute_pairwise_lipschitz_matrix, compute_pairwise_lipschitz_vector,
    compute_pairwise_lipschitz_tensor, estimate_lipschitz_probability)
from alpha_gradient.statistical_analysis import (
    compute_confidence_interval, compute_confidence_probability)

import warnings 
warnings.filterwarnings("error")

heaviside = BallWithWall()

sweep = 100
xspace = np.linspace(0, np.pi/2, sweep)
eps_lst_tight = np.zeros((sweep,2))
eps_lst_loose = np.zeros((sweep,2))
obj_lst = np.zeros(sweep)
fobg_value_lst = np.zeros(sweep)
zobg_value_lst = np.zeros(sweep)
n_samples = 1000
sigma = 0.05

for i in range(sweep):
    x = xspace[i] * np.ones(1)
    fobg_mu, fobg_sigma = heaviside.first_order_batch_gradient(x, n_samples, sigma)
    zobg_mu, zobg_sigma = heaviside.zero_order_batch_gradient(x, n_samples, sigma)

    obj_lst[i] = heaviside.evaluate(x, np.zeros(1))
    fobg_value_lst[i] = fobg_mu
    zobg_value_lst[i] = zobg_mu

    ci_tight = compute_confidence_interval(
        zobg_mu, zobg_sigma ** 2.0, 1000, 10.0, 0.95)
    ci_loose = compute_confidence_interval(
        zobg_mu, zobg_sigma ** 2.0, 100, 10.0, 0.95)

    compute_confidence_probability

    bias = fobg_mu - zobg_mu
    ci_tight_range = ci_tight[0] - ci_tight[1]
    ci_loose_range = ci_loose[0] - ci_loose[1]

    print(bias)

    eps_lst_tight[i,0] = ci_tight[0]
    eps_lst_tight[i,1] = ci_tight[1]    
    eps_lst_loose[i,0] = ci_loose[0]
    eps_lst_loose[i,1] = ci_loose[1]    

plt.figure()
plt.subplot(2,1,2)
plt.plot(xspace, obj_lst, 'k-')
plt.subplot(2,1,1)
plt.plot(xspace, zobg_value_lst, 'k-')
plt.plot(xspace, fobg_value_lst, 'r-')
plt.gca().fill_between(
    xspace, zobg_value_lst + eps_lst_loose[:,0],
    zobg_value_lst + eps_lst_loose[:,1], alpha=0.2,
    color='blue', label='100 Samples')    
plt.gca().fill_between(
    xspace, zobg_value_lst + eps_lst_tight[:,0],
    zobg_value_lst + eps_lst_tight[:,1], alpha=0.2,
    color='red', label='1000 Samples')
plt.legend()
plt.show()

    
    
    