import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 22})

from tqdm import tqdm
from soft_heaviside import SoftHeaviside

plt.figure()

grid_size = 1000
obj = SoftHeaviside(0.01)

#=============================================================================

sample_size = 10

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

"""
plt.subplot(2,2,2)
plt.plot(x_range, obj_storage, 'k-',
    label=r'$f(x,0), \Delta = 0.5$')
plt.legend()
"""

plt.subplot(2,2,3)
plt.plot(x_range, zobg_storage, 'b-',
    label=r'$\hat{\nabla}^0_x F(x), N=1$')
plt.plot(x_range, fobg_storage, 'r-',
    label=r'$\hat{\nabla}^1_x F(x), N=1$')
plt.legend()

plt.subplot(2,2,4)
plt.plot(x_range, zobv_storage, 'b-',
    label=r'Var$(\hat{\nabla}^0_x F(x)), N=1$')
plt.plot(x_range, fobv_storage, 'r-',
    label=r'Var$(\hat{\nabla}^1_x F(x)), N=1$')

plt.legend()

#=============================================================================

#=============================================================================

sample_size = 100
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
plt.subplot(2,2,3)
plt.plot(x_range, zobg_storage, '-',
    label=r'$\hat{\nabla}^0_x F(x), \Delta = 0.01$', color='skyblue')
plt.plot(x_range, fobg_storage, '-',
    label=r'$\hat{\nabla}^1_x F(x), \Delta = 0.01$', color='pink')
plt.legend()

plt.subplot(2,2,4)
plt.plot(x_range, zobv_storage, '-',
    label=r'Var$(\hat{\nabla}^0_x F(x)), \Delta=0.01$', color='skyblue')
plt.plot(x_range, fobv_storage, '-',
    label=r'Var$(\hat{\nabla}^1_x F(x)), \Delta=0.01$', color='pink')

plt.legend()

#=============================================================================
plt.show()
