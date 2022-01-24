import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 20})

from tqdm import tqdm
from soft_heaviside import SoftHeaviside

plt.figure()

grid_size = 1000
obj = SoftHeaviside(0.001)
bobj = SoftHeaviside(0.2)
x_range = np.linspace(-1, 1, grid_size)
obj_storage = np.zeros(grid_size)
bobj_storage = np.zeros(grid_size)

for i in tqdm(range(grid_size)):
    x = x_range[i] * np.ones(1)
    obj_storage[i] = obj.evaluate(x, np.zeros(1))
    bobj_storage[i] = bobj.evaluate(x, np.zeros(1))


plt.subplot(2,2,2)
plt.plot(x_range, obj_storage, 'k-', label='Coulomb Friction')
plt.plot(x_range, bobj_storage, color='springgreen', label='Approximation')
plt.xlabel('Tangential velocity')
plt.ylabel('Frictional force')
plt.legend()


#=============================================================================
obj = SoftHeaviside(0.005)
stdev = 0.1
x_range = np.linspace(-1, 1, grid_size)
obj_storage = np.zeros(grid_size)
bobj_storage = np.zeros(grid_size)

fobg_storage = np.zeros(grid_size)
zobg_storage = np.zeros(grid_size)

fobv_storage = np.zeros(grid_size)
zobv_storage = np.zeros(grid_size)

for i in tqdm(range(grid_size)):
    x = 0.1 * np.ones((1))

    obj_storage[i] = obj.evaluate(x, np.zeros(1))
    #bobj_storage[i], _ = obj.bundled_objective(x, i+1, stdev)

    fobg, fobv = obj.first_order_batch_gradient(
        x, i+1, stdev)
    zobg, zobv = obj.zero_order_batch_gradient(
        x, i+1, stdev) 

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
plt.plot(range(grid_size), fobg_storage, 'r-',
    label='FOBG')
plt.plot(range(grid_size), zobg_storage, 'b-',
    label='ZOBG')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Gradient Estimate')
plt.legend()

#=============================================================================

#=============================================================================

sample_size = 1000
nu_range = np.logspace(-5, -1, grid_size)
obj_storage = np.zeros(grid_size)
bobj_storage = np.zeros(grid_size)

fobg_storage = np.zeros(grid_size)
zobg_storage = np.zeros(grid_size)

fobv_storage = np.zeros(grid_size)
zobv_storage = np.zeros(grid_size)

for i in tqdm(range(grid_size)):
    obj = SoftHeaviside(nu_range[i])    
    x = np.zeros(1)

    #obj_storage[i] = obj.evaluate(x, np.zeros(1))
    #bobj_storage[i], _ = obj.bundled_objective(x, sample_size, stdev)

    fobg, fobv = obj.first_order_batch_gradient(
        x, sample_size, stdev)
    zobg, zobv = obj.zero_order_batch_gradient(
        x, sample_size, stdev) 

    fobg_storage[i] = fobg
    fobv_storage[i] = fobv
    zobg_storage[i] = zobg
    zobv_storage[i] = zobv

#=============================================================================
plt.subplot(2,2,4)
plt.plot(np.log10(nu_range), np.log10(zobv_storage), '-',
    label='ZOBG', color='blue')
plt.plot(np.log10(nu_range), np.log10(fobv_storage), '-',
    label='FOBG', color='red')
plt.xlabel('Stiction Velocity (log-scale)')
plt.ylabel('Empirical Variance (log-scale)')
plt.legend()

#=============================================================================
plt.show()
