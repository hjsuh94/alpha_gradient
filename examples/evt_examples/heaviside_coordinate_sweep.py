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

from scipy.optimize import fsolve

import warnings 
warnings.filterwarnings("error")

heaviside = HeavisideAllPositive(1)

sweep = 100
xspace = np.linspace(-1, 1, sweep)
Lbar = 20
alpha_lst = np.zeros(sweep)
value_lst = np.zeros(sweep)

for i in tqdm(range(len(xspace))):
    mean = xspace[i]
    sigma = 0.1
    trials = 1000
    subbatch_size = 5

    X = np.random.normal(mean, sigma, (trials, subbatch_size, 1))
    X_flatten = X.reshape((trials * subbatch_size, 1))
    y_flatten = 0.5 * heaviside.evaluate_batch(np.array([0]), X_flatten)
    y = y_flatten.reshape((trials, subbatch_size, 1))    

    alpha_lst[i] = estimate_lipschitz_probability(X, y, Lbar)
    value_lst[i] = heaviside.evaluate(np.array([mean]), np.zeros(1))

plt.figure()
plt.plot(xspace, alpha_lst, 'r-')
plt.plot(xspace, value_lst, 'k-')
plt.show()

