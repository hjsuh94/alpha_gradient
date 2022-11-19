import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt

def compute_conditional(sigma, width):
    """
    Compute the conditional expectation E[f(x) | e_width]
    = E[f(x) * ind(e_width)] / Pr(e_width)
    """
    conditional = (1 - norm.cdf(width / 2, scale=sigma)) + (norm.cdf(
        -width / 2, scale=sigma) - 0)
    prob = (2 - width) / 2
    return conditional / prob

eps = 0.1

eps_storage = np.linspace(0.0001,0.5,1000)
delta_storage = np.zeros((10,1000))
sigma_storage = np.linspace(0.001, 0.1, 10)
var_storage = np.zeros((10,1000))
empirical_var_storage = np.zeros(10)

for j in range(10):
    sigma = sigma_storage[j]
    for i in range(1000):
        eps = eps_storage[i]
        beta = eps / 2
        delta = np.abs(compute_conditional(sigma, eps) - 1)
        delta_storage[j,i] = delta
        var_storage[j,i] = (((1 - beta) * delta - beta) ** 2.0) / beta

    samples = -1 + 2.0 * np.random.rand(10000000)
    mean = np.mean(norm.pdf(samples, scale=sigma))
    empirical_var = np.var(norm.pdf(samples, scale=sigma))
    print("Sigma:" + str(sigma))
    print("Mean: " + str(mean))
    print("Var:  " + str(empirical_var))
    print('==========================')

    empirical_var_storage[j] = empirical_var

eps_storage = eps_storage / 2

plt.figure()
plt.plot(empirical_var_storage)
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.plot(eps_storage, delta_storage[0,:],'r-')
plt.plot(eps_storage, delta_storage[1,:],'g-')
plt.plot(eps_storage, delta_storage[2,:],'b-')
plt.xlabel('beta')
plt.ylabel('delta')

plt.subplot(1,2,2)
plt.plot(eps_storage, var_storage[0,:],'r-')
plt.plot(eps_storage, var_storage[1,:],'g-')
plt.plot(eps_storage, var_storage[2,:],'b-')
plt.xlabel('beta')
plt.ylabel('delta')
plt.show()
