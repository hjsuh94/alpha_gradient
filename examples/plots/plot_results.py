import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from alpha_gradient.plotting_tools import plot_cost

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 22})

plt.figure()


cost_fobg = np.load("data/curling2/curl_fobg_1000_cost.npy")
cost_zobg = np.load("data/curling2/curl_zobg_1000_cost.npy")
cost_aobg = np.load("data/curling2/curl_aobg_1000_cost.npy")

plt.subplot(2,2,1)
#plt.title('Pushing, k=1000.0')
plot_cost(plt.gca(), np.log(cost_fobg[:500]), 4, 'red', label='FOBG')
plot_cost(plt.gca(), np.log(cost_zobg[:500]), 4, 'blue', label='ZOBG')
plot_cost(plt.gca(), np.log(cost_aobg[:500]), 4, 'springgreen', label='AOBG')
plt.xlabel('iterations')
plt.ylabel('cost (log-scale)')
plt.legend()

cost_fobg = np.load("data/curling2/curl_fobg_10_cost.npy")
cost_zobg = np.load("data/curling2/curl_zobg_10_cost.npy")
cost_aobg = np.load("data/curling2/curl_aobg_10_cost.npy")

plt.subplot(2,2,3)
#plt.title('Pushing, k=10.0')
plot_cost(plt.gca(), np.log(cost_zobg), 4, 'blue', label='ZOBG')
plot_cost(plt.gca(), np.log(cost_aobg), 4, 'springgreen', label='AOBG')
plot_cost(plt.gca(), np.log(cost_fobg), 4, 'red', label='FOBG', style='--')
plt.xlabel('iterations')
plt.ylabel('cost (log-scale)')
plt.legend()

cost_fobg = np.load("data/friction/friction_fobg_cost.npy")
cost_zobg = np.load("data/friction/friction_zobg_cost.npy")
cost_aobg = np.load("data/friction/friction_aobg_cost.npy")

plt.subplot(2,2,2)
#plt.title('Friction')
plot_cost(plt.gca(), np.log(cost_zobg), 4, 'blue', label='ZOBG')
plot_cost(plt.gca(), np.log(cost_aobg), 4, 'springgreen', label='AOBG')
plot_cost(plt.gca(), np.log(cost_fobg), 4, 'red', label='FOBG')
plt.xlabel('iterations')
plt.ylabel('cost (log-scale)')
plt.legend()

cost_fobg = np.load("data/breakout/fobg_cost.npy")
cost_zobg = np.load("data/breakout/zobg_cost.npy")
cost_aobg = np.load("data/breakout/aobg_cost.npy")

plt.subplot(2,2,4)
#plt.title('Tennis')
plot_cost(plt.gca(), np.log(cost_zobg), 4, 'blue', label='ZOBG')
plot_cost(plt.gca(), np.log(cost_aobg), 4, 'springgreen', label='AOBG')
plot_cost(plt.gca(), np.log(cost_fobg), 4, 'red', label='FOBG')
plt.xlabel('iterations')
plt.ylabel('cost (log-scale)')
plt.legend()

plt.show()

