import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 22})

from tqdm import tqdm
from heaviside_torch import HeavisideTorch
from ball_with_wall_torch import BallWithWallTorch
from angular_torch import AngularTorch

plt.figure()

grid_size = 1000
sample_size = 1000

#=============================================================================
plt.subplot(1,3,1)
obj = HeavisideTorch()

stdev = 0.3
x = np.linspace(-np.pi/2,np.pi/2,grid_size)
obj_storage = np.zeros(grid_size)
bobj_storage = np.zeros(grid_size)

for i in tqdm(range(grid_size)):
    obj_storage[i] = obj.evaluate(np.array([x[i]]), np.zeros(1))
    bobj_storage[i], _ = obj.bundled_objective(
        np.array([x[i]]), sample_size, stdev)

plt.plot(x, obj_storage, 'k-', label=r'$f(x,0)$')
plt.plot(x, bobj_storage, 'r-', label=r'$F(x)$')
plt.xticks([]),plt.yticks([])
plt.xlabel(r'$\theta$')
#plt.legend()

#=============================================================================

plt.subplot(1,3,2)

obj = BallWithWallTorch()

stdev = 0.07
x = np.linspace(0,np.pi/2,grid_size)
obj_storage = np.zeros(grid_size)
bobj_storage = np.zeros(grid_size)

for i in tqdm(range(grid_size)):
    obj_storage[i] = obj.evaluate(np.array([x[i]]), np.zeros(1))
    bobj_storage[i], _ = obj.bundled_objective(
        np.array([x[i]]), sample_size, stdev)

plt.plot(x, obj_storage, 'k-', label=r'$f(x,0)$')
plt.plot(x, bobj_storage, 'r-', label=r'$F(x)$')
plt.xticks([]),plt.yticks([])
plt.xlabel(r'$\theta$')
#plt.legend()

#=============================================================================

plt.subplot(1,3,3)

obj = AngularTorch()

stdev = 0.05
x = np.linspace(-np.pi/3,np.pi/3,grid_size)
obj_storage = np.zeros(grid_size)
bobj_storage = np.zeros(grid_size)

for i in tqdm(range(grid_size)):
    obj_storage[i] = obj.evaluate(np.array([x[i]]), np.zeros(1))
    bobj_storage[i], _ = obj.bundled_objective(
        np.array([x[i]]), sample_size, stdev)

plt.plot(x, obj_storage, 'k-', label=r'$f(x,0)$')
plt.plot(x, bobj_storage, 'r-', label=r'$F(x)$') 
plt.xticks([]),plt.yticks([])
plt.xlabel(r'$\theta$')
#plt.legend()

#=============================================================================

plt.show()
