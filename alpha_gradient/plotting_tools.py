import numpy as np
import time 
import matplotlib.pyplot as plt 

from alpha_gradient.statistical_analysis import (
    compute_mean, compute_variance_norm)

def plot_cost(ax, cost_array, sma_window, color, label, style='-'):
    smoothed_cost = np.zeros(len(cost_array))
    smoothed_var = np.zeros(len(cost_array))    
    for i in range(len(cost_array)):
        window_start = np.max([0, i - sma_window ])
        window_end = np.min([len(cost_array), i + sma_window])
        window = cost_array[window_start:window_end]

        smoothed_cost[i] = compute_mean(window)
        smoothed_var[i] = np.sqrt(compute_variance_norm(window))

    ax.plot(smoothed_cost, color=color, alpha=1.0, label=label, linestyle=style)
    ax.plot(cost_array, color=color, alpha=0.1)    
    ax.fill_between(range(len(cost_array)),
        smoothed_cost - smoothed_var, smoothed_cost + smoothed_var,
        color=color, alpha=0.2)

def plot_data(ax, x_array, cost_array, sma_window, color, label, style='-'):
    smoothed_cost = np.zeros(len(cost_array))
    smoothed_var = np.zeros(len(cost_array))    
    for i in range(len(cost_array)):
        window_start = np.max([0, i - sma_window ])
        window_end = np.min([len(cost_array), i + sma_window])
        window = cost_array[window_start:window_end]

        smoothed_cost[i] = compute_mean(window)
        smoothed_var[i] = np.sqrt(compute_variance_norm(window))

    ax.plot(x_array, smoothed_cost, color=color, alpha=1.0, 
        label=label, linestyle=style)
    ax.plot(x_array, cost_array, color=color, alpha=0.1)    
    ax.fill_between(x_array,
        smoothed_cost - smoothed_var, smoothed_cost + smoothed_var,
        color=color, alpha=0.2)

def plot_data_log(ax, x_array, cost_array, sma_window, color, label, style='-'):
    smoothed_cost = np.zeros(len(cost_array))
    smoothed_var = np.zeros(len(cost_array))    
    for i in range(len(cost_array)):
        window_start = np.max([0, i - sma_window ])
        window_end = np.min([len(cost_array), i + sma_window])
        window = cost_array[window_start:window_end]

        smoothed_cost[i] = compute_mean(window)
        smoothed_var[i] = np.sqrt(compute_variance_norm(window))

    ax.plot(x_array, np.log(smoothed_cost), color=color, alpha=1.0, 
        label=label, linestyle=style)
    ax.plot(x_array, np.log(cost_array), color=color, alpha=0.1)    
    ax.fill_between(x_array, range(len(cost_array)),
        np.log(smoothed_cost - smoothed_var),
        np.log(smoothed_cost + smoothed_var),
        color=color, alpha=0.2)

