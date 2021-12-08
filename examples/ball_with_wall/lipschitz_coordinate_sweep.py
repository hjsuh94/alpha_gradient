import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.statistical_analysis import compute_mean, compute_variance_norm
from alpha_gradient.lipschitz_estimator import estimate_lipschitz_probability
from ball_with_wall import BallWithWall

objective = BallWithWall()

sweep = 100
xspace = np.linspace(0, np.pi/2, sweep)
Lbar = 0.1
alpha_lst = np.zeros(sweep)
value_lst = np.zeros(sweep)

for i in tqdm(range(len(xspace))):
    mean = xspace[i]
    sigma = 0.1
    trials = 1000
    subbatch_size = 3

    X = np.random.normal(mean, sigma, (trials, subbatch_size, 1))
    X_flatten = X.reshape((trials * subbatch_size, 1))
    y_flatten = objective.evaluate_batch(np.array([0]), X_flatten)
    y = y_flatten.reshape((trials, subbatch_size, 1))    

    alpha_lst[i] = estimate_lipschitz_probability(X, y, Lbar)
    value_lst[i] = objective.evaluate(np.array([mean]), np.zeros(1))

normalized_value_lst = (value_lst - np.min(value_lst)) / (
    np.max(value_lst) - np.min(value_lst))

plt.figure()
plt.plot(xspace, alpha_lst, 'r-')
plt.plot(xspace, normalized_value_lst, 'k-')
plt.show()
