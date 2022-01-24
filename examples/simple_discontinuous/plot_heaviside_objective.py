import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 30})

from tqdm import tqdm
from heaviside_objective import HeavisideAllPositive

plt.figure()

grid_size = 1000
sample_size = 1000

#=============================================================================

obj = HeavisideAllPositive(1)

stdev = 0.1
x_range = np.linspace(-1, 1, grid_size)
obj_storage = np.zeros(grid_size)
bobj_storage = np.zeros(grid_size)

fobg_storage = np.zeros(grid_size)
zobg_storage = np.zeros(grid_size)

fobv_storage = np.zeros(grid_size)
zobv_storage = np.zeros(grid_size)

for i in tqdm(range(grid_size)):
    x = np.array([x_range[i]])

    obj_storage[i] = obj.evaluate(x, np.zeros(1))
    bobj_storage[i], _ = obj.bundled_objective(x, sample_size, stdev)

    fobg, fobv = obj.first_order_batch_gradient(
        x, sample_size, stdev)
    zobg, zobv = obj.zero_order_batch_gradient(
        x, sample_size, stdev) 

    fobg_storage[i] = fobg
    fobv_storage[i] = fobv
    zobg_storage[i] = zobg
    zobv_storage[i] = zobv

#=============================================================================

plt.subplot(1,3,1)
plt.plot(x_range, obj_storage, 'k-', label=r'$f(\theta,w)$')
plt.plot(x_range, bobj_storage, 'r-', label=r'$F(\theta)$')
plt.legend()

plt.subplot(1,3,2)
plt.plot(x_range, zobg_storage, 'b-', label=r'$\hat{\nabla}^0_\theta F(\theta)$')
plt.plot(x_range, fobg_storage, 'r-', label=r'$\hat{\nabla}^1_\theta F(\theta)$')
plt.legend()

plt.subplot(1,3,3)
plt.plot(x_range, zobv_storage, 'b-',
    label=r'Var$(\hat{\nabla}^0_\theta F(\theta))$')
plt.plot(x_range, fobv_storage, 'r-',
    label=r'Var$(\hat{\nabla}^1_\theta F(\theta))$')

plt.legend()
plt.show()
