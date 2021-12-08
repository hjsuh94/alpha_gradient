import numpy as np
import matplotlib.pyplot as plt


mean = 0.0
sigma = 1.0
samples = 100000
samples_lst = np.zeros(samples)
subsamples = 10000

for t in range(samples):
    samples_lst[t] = np.max(np.random.normal(mean, sigma, subsamples))

plt.figure()
hist = np.histogram(samples_lst, bins=100)
print(hist[0])
print(hist[1])
plt.hist(samples_lst, bins=hist[1])
plt.show()