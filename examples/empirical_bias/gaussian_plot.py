import numpy as np
from scipy.stats import norm 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 22})

sigma_bar = 0.01
sigma_samples = np.linspace(0.04, 0.2, 5)

print(sigma_samples)
domain = np.linspace(-0.7, 0.7, 10000)
beta = 0.06

colormap = cm.get_cmap('cool')
plt.figure(figsize=(14,10))

#%%
plt.subplot(2,2,1)

def plot_cdf(domain, sigma, color, alpha, label=None):
    values = norm.cdf(domain, scale=sigma)
    plt.plot(domain, values, color=color, alpha=alpha, label=label)

def plot_event(domain, sigma, minmax, color, alpha):
    #values[np.logical_and((domain <0.5), (domain > 0.5 * beta))] = -0.1
    #values[np.logical_and((domain >-0.5), (domain < -0.5 * beta))] = -0.1

    values_minus = (norm.cdf((domain + 0.5) / sigma) - norm.cdf(
        (domain + 0.5 * beta)/ sigma)) / (0.5 - 0.5 * beta)

    values_plus = -(norm.cdf((domain - 0.5) / sigma) - norm.cdf(
        (domain - 0.5 * beta)/ sigma)) / (0.5 - 0.5 * beta)        

    values = values_minus + values_plus

    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    values = (minmax[1] - minmax[0]) * values + minmax[0]

    plt.plot(domain, values, color=color, alpha=alpha)

plot_cdf(domain, sigma_bar, 'black', 1.0)
plot_event(domain, 0.001, [-0.2, -0.1], 'black', 1.0)
for i in range(len(sigma_samples)):
    sigma = np.sqrt(sigma_bar ** 2.0 + sigma_samples[i] ** 2.0)
    plot_cdf(domain, sigma, colormap((i + 1) / len(sigma_samples)), 0.8,
        label=r'$\sigma={:.2f}$'.format(sigma_samples[i]))
    plot_event(domain, sigma, [-0.2, -0.1], colormap(
        (i+1)/len(sigma_samples)), 0.8)

plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')
plt.legend(loc='upper left')

#%% 
plt.subplot(2,2,2)

def plot_pdf(domain, sigma, color, alpha):
    values = norm.pdf(domain, scale=sigma)
    plt.plot(domain, values, color=color, alpha=alpha)

plot_pdf(domain, sigma_bar, 'black', 1.0)
plot_event(domain, 0.001, [-8, -4], 'black', 1.0)
for i in range(len(sigma_samples)):
    sigma = np.sqrt(sigma_bar ** 2.0 + sigma_samples[i] ** 2.0)
    plot_pdf(domain, sigma, colormap((i + 1) / len(sigma_samples)), 0.8)
    plot_event(domain, sigma, [-8, -4], colormap(
        (i+1)/len(sigma_samples)), 0.8)

plt.xlabel(r'$x$')
plt.ylabel(r'$\nabla^1_x f(x)$')


#%% 
plt.subplot(2,2,3)

def get_expected(sigma, num_samples):
    samples = np.random.rand(num_samples) - 0.5
    # reject samples outside the event.
    samples_rejected = samples[
        np.logical_or((samples < -0.5 * beta), (samples > 0.5 * beta))]
    return np.mean(norm.pdf(samples_rejected, scale=sigma))

def get_full(sigma):
    return norm.cdf(0.5, scale=sigma) - norm.cdf(-0.5, scale=sigma)

sigma_space = np.linspace(sigma_bar, 0.2, 1000)
delta_space = np.zeros(len(sigma_space))
for i in range(len(sigma_space)):
    delta_space[i] = np.abs(get_expected(
        sigma_space[i], 100000) - get_full(sigma_space[i]))
plt.plot(sigma_space, delta_space, 'k-')    

delta_bar = np.abs(get_expected(
    sigma_bar, 100000) - get_full(sigma_bar))
plt.plot(sigma_bar, delta_bar, marker='o', color='black', alpha=0.8,
    markersize=10)

delta_samples = np.zeros(len(sigma_samples))
for i in range(len(sigma_samples)):
    delta_samples[i] = np.abs(get_expected(
        sigma_samples[i], 100000) - get_full(sigma_space[i]))
    plt.plot(sigma_samples[i], delta_samples[i], marker='o', color=colormap(
        (i+1)/len(sigma_samples)), alpha=0.8, markersize=10)
plt.xlabel(r'$\sigma$ (Added Smoothing)')
plt.ylabel(r'$\Delta$')

plt.subplot(2,2,4)

def get_variance(sigma, num_samples):
    samples = np.random.rand(num_samples) - 0.5
    return np.var(norm.pdf(samples, scale=sigma))

sigma_space = np.linspace(sigma_bar, 0.2, 1000)
var_space = np.zeros(len(sigma_space))
for i in range(len(sigma_space)):
    var_space[i] = np.log(get_variance(sigma_space[i], 100000))
plt.plot(sigma_space, var_space, 'k-')    

plt.plot(sigma_bar, np.log(get_variance(
    sigma_bar, 100000)), marker='o', color='black', alpha=0.8,
    markersize=10)
var_samples = np.zeros(len(sigma_samples))
for i in range(len(sigma_samples)):
    var_samples[i] = np.log(get_variance(sigma_samples[i], 100000))
    plt.plot(sigma_samples[i], var_samples[i], marker='o', color=colormap(
        (i+1)/len(sigma_samples)), alpha=0.8, markersize=10)

plt.xlabel(r'$\sigma$ (Added Smoothing)')
plt.ylabel(r'$\log \mathsf{Var}[\nabla^1_x f(x)]$')
        
plt.savefig("smoothing_fig.png")
plt.show()
