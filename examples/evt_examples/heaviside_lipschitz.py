import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time



from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction
from heaviside_objective import HeavisideAllPositive
from alpha_gradient.lipschitz_estimator import (
    compute_pairwise_lipschitz_matrix, compute_pairwise_lipschitz_vector,
    compute_pairwise_lipschitz_tensor)

from scipy.optimize import fsolve

import warnings 
warnings.filterwarnings("error")

heaviside = HeavisideAllPositive(1)

def psi(alpha, x):
    """
    Given scalar alpha and a (B) array of x, compute the psi function.
    """
    # Replace zeros with a small constant here for numerical reasons.
    x[x == 0] = 1e-3
    k = len(x)
    a = 1 / alpha 
    b = (1/k) * np.sum(np.power(x, -alpha) * np.log(x))
    c = (1/k) * np.sum(np.power(x, -alpha))
    d = (1/k) * np.sum(np.log(x))
    return a + b / c - d

def sigma(alpha, x):
    """
    Given scalar alpha and a (B) array of x, compute the sigma function.
    """
    k = len(x)
    return np.power((1/k) * np.sum(np.power(x, -alpha)), -1/alpha)

def plot_frechet(alpha, sigma, x):
    a = alpha / sigma
    b = np.power(x / sigma, -1 - alpha)
    c = np.exp(-np.power((x/sigma), -alpha))
    return a * b * c

def compute_cdf(alpha, sigma, x):
    return np.exp(-np.pow(x/sigma, -alpha))

###################

# Lipschitz constant estimation.
mean = 0.0
std = 1.0
samples = 1000
trials = 1000
trial_lst = np.zeros(trials)
subsamples = 100

start_time = time.time()
x = np.random.normal(mean, std, (samples, 1))
y = heaviside.evaluate_batch(np.array([0]), x)
samples_range = range(samples)

for t in tqdm(range(trials)):
    ind = np.random.choice(samples_range, subsamples, replace=False)
    x_sample = x[ind]
    y_sample = y[ind]
    trial_lst[t] = np.max(compute_pairwise_lipschitz_vector(x_sample, y_sample))
print(time.time() - start_time)

subbatch_size = 1
start_time = time.time()
x = np.random.normal(mean, std, (trials, subbatch_size, 1))
x_flatten = x.reshape((trials * subbatch_size, 1))
y_flatten = heaviside.evaluate_batch(np.array([0]), x_flatten)
y = y_flatten.reshape((trials, subbatch_size, 1))
trial_mat = compute_pairwise_lipschitz_tensor(x, y)
trial_lst = np.max(trial_mat, axis=1)
print(trial_lst.shape)
print(time.time() - start_time)


start_time = time.time()
alpha_star = fsolve(lambda alpha: psi(alpha, trial_lst), 0.1)
sigma_star = sigma(alpha_star, trial_lst)
print(time.time() - start_time)
print(alpha_star, sigma_star)


hist = np.histogram(trial_lst, bins=500)
plt.figure(figsize=(10,9))
plt.subplot(3,1,1)
plt.title("x: " + str(mean))
plt.hist(trial_lst, bins=hist[1], color='b', label='Distribution of Samples')
frechet_lst = plot_frechet(alpha_star, sigma_star, hist[1])
plt.plot(hist[1], np.max(hist[0]) / np.max(frechet_lst) * frechet_lst,
    color='r', label='Fitted Frechet Distribution')
plt.xlim([-20, 150])
plt.legend()

###################

# Lipschitz constant estimation.
mean = 1.0
std = 1.0
samples = 1000
trials = 1000
trial_lst = np.zeros(trials)
subsamples = 100

start_time = time.time()
x = np.random.normal(mean, std, (samples, 1))
y = heaviside.evaluate_batch(np.array([0]), x)
samples_range = range(samples)

for t in tqdm(range(trials)):
    ind = np.random.choice(samples_range, subsamples, replace=False)
    x_sample = x[ind]
    y_sample = y[ind]
    trial_lst[t] = np.max(compute_pairwise_lipschitz_vector(x_sample, y_sample))
print(time.time() - start_time)

subbatch_size = 1
start_time = time.time()
x = np.random.normal(mean, std, (trials, subbatch_size, 1))
x_flatten = x.reshape((trials * subbatch_size, 1))
y_flatten = heaviside.evaluate_batch(np.array([0]), x_flatten)
y = y_flatten.reshape((trials, subbatch_size, 1))
trial_mat = compute_pairwise_lipschitz_tensor(x, y)
trial_lst = np.max(trial_mat, axis=1)
print(trial_lst.shape)
print(time.time() - start_time)


start_time = time.time()
alpha_star = fsolve(lambda alpha: psi(alpha, trial_lst), 0.1)
sigma_star = sigma(alpha_star, trial_lst)
print(time.time() - start_time)
print(alpha_star, sigma_star)


hist = np.histogram(trial_lst, bins=500)
plt.subplot(3,1,2)
plt.title("x: " + str(mean))
plt.hist(trial_lst, bins=hist[1], color='b', label='Distribution of Samples')
frechet_lst = plot_frechet(alpha_star, sigma_star, hist[1])
plt.plot(hist[1], np.max(hist[0]) / np.max(frechet_lst) * frechet_lst,
    color='r', label='Fitted Frechet Distribution')
plt.xlim([-20, 150])
plt.legend()

###################

# Lipschitz constant estimation.
mean = 2.0
std = 1.0
samples = 1000
trials = 1000
trial_lst = np.zeros(trials)
subsamples = 100

start_time = time.time()
x = np.random.normal(mean, std, (samples, 1))
y = heaviside.evaluate_batch(np.array([0]), x)
samples_range = range(samples)

for t in tqdm(range(trials)):
    ind = np.random.choice(samples_range, subsamples, replace=False)
    x_sample = x[ind]
    y_sample = y[ind]
    trial_lst[t] = np.max(compute_pairwise_lipschitz_vector(x_sample, y_sample))
print(time.time() - start_time)

subbatch_size = 1
start_time = time.time()
x = np.random.normal(mean, std, (trials, subbatch_size, 1))
x_flatten = x.reshape((trials * subbatch_size, 1))
y_flatten = heaviside.evaluate_batch(np.array([0]), x_flatten)
y = y_flatten.reshape((trials, subbatch_size, 1))
trial_mat = compute_pairwise_lipschitz_tensor(x, y)
trial_lst = np.max(trial_mat, axis=1)
print(trial_lst.shape)
print(time.time() - start_time)


start_time = time.time()
alpha_star = fsolve(lambda alpha: psi(alpha, trial_lst), 0.1)
sigma_star = sigma(alpha_star, trial_lst)
print(time.time() - start_time)
print(alpha_star, sigma_star)


hist = np.histogram(trial_lst, bins=500)
plt.subplot(3,1,3)
plt.title("x: " + str(mean))
plt.hist(trial_lst, bins=hist[1], color='b', label='Distribution of Samples')
frechet_lst = plot_frechet(alpha_star, sigma_star, hist[1])
plt.plot(hist[1], np.max(hist[0]) / np.max(frechet_lst) * frechet_lst,
    color='r', label='Fitted Frechet Distribution')
plt.xlim([-20, 150])
plt.legend()


plt.show()

