import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from tqdm import tqdm
import torch

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.smoothing import (relu, relu_batch,
    softplus, softplus_batch)

class PivotSoftPlus(DynamicalSystem):
    def __init__(self):
        super().__init__()
        self.h = 0.005
        self.dim_x = 6
        self.dim_u = 0

        self.d = 1
        self.v0 = 2.0
        self.T = 200

        self.stiffness = np.power(10,4)
        self.kappa = 100

        # Geometrical parameters
        self.width = 0.05
        self.height = 0.5
        self.dist = 0.7
        self.radius = 0.03

    def calc_rotation_matrix(self, theta_batch):
        """
        Compute rotation matrix
        """
        B = theta_batch.shape[0]
        rotmat_batch = torch.zeros((B, 2, 2))
        rotmat_batch[:,0,0] = torch.cos(theta_batch)
        rotmat_batch[:,0,1] = torch.sin(theta_batch)
        rotmat_batch[:,1,1] = torch.cos(theta_batch)
        rotmat_batch[:,1,0] = -torch.sin(theta_batch)
        return rotmat_batch

    def calc_transformed_ball_coordinates(self, x_batch):
        """
        Given angle theta and ball's radius, compute angle.
        """
        theta_batch = x_batch[:,2]
        R_wt_batch = self.calc_rotation_matrix(theta_batch)
        R_tw_batch = R_wt_batch.transpose(1,2)
        xy_batch = x_batch[:,0:2]
        xy_batch_centered = xy_batch - torch.Tensor([self.dist, 0.0])
        xy_rotated_centered = torch.einsum("bij,bj->bi", R_tw_batch, xy_batch_centered)
        return xy_rotated_centered 
    
    def calc_sg_batch(self, x_batch):
        xy_rotated_centered = self.calc_transformed_ball_coordinates(x_batch)
        B = x_batch.shape[0]
        dists = torch.zeros(B)
        normals = torch.zeros(B, 2)
        witnesses = torch.zeros(B, 2)

        xn = xy_rotated_centered[:,0]
        yn = xy_rotated_centered[:,1]
        
        ind = torch.logical_and(torch.logical_and((xn < 0), (yn < self.height)), (yn > 0))
        normals[ind,:] = torch.Tensor([-1.0, 0.0])
        dists[ind] = -xn[ind] - self.radius - self.width / 2
        witnesses[ind,0] = -self.width/2
        witnesses[ind,1] = yn[ind]

        ind = torch.logical_and(torch.logical_and((xn > 0), (yn < self.height)), (yn > 0))
        normals[ind,:] = torch.Tensor([1.0, 0.0])
        dists[ind] = xn[ind] - self.radius - self.width / 2
        witnesses[ind,0] = self.width/2
        witnesses[ind,1] = yn[ind]

        ind = yn > self.height
        center = torch.Tensor([0.0, self.height])
        dist_to_center = torch.linalg.norm(xy_rotated_centered[ind,:] - center, dim=1)
        dists[ind] = dist_to_center - self.radius - self.width / 2
        normals[ind,:] = (xy_rotated_centered[ind,:] - center) / dist_to_center[:,None]
        witnesses[ind,:] = center + normals[ind,:] * self.width/2

        ind = yn < 0.0
        center = torch.Tensor([0.0, 0.0])
        dist_to_center = torch.linalg.norm(xy_rotated_centered[ind,:] - center, dim=1)
        dists[ind] = dist_to_center - self.radius - self.width / 2
        normals[ind,:] = (xy_rotated_centered[ind,:] - center) / dist_to_center[:,None]
        witnesses[ind,:] = center + normals[ind,:] * self.width/2        

        theta_batch = x_batch[:,2]
        R_wt_batch = self.calc_rotation_matrix(theta_batch)

        normals_W = torch.einsum("Bij,Bj->Bi", R_wt_batch, normals)
        witnesses_W = torch.einsum("Bij,Bj->Bi", R_wt_batch, witnesses) + torch.Tensor(
            [self.dist, 0.0])

        return dists, normals_W, witnesses_W, normals, witnesses
    
    def dynamics_batch(self, x_batch):
        p_now = x_batch[:,0:2].clone()
        theta_now = x_batch[:,2].clone()
        v_now = x_batch[:,3:5].clone()
        omega_now = x_batch[:,5].clone()

        # Compute forces
        dists, normals_W, witnesses_W, normals, witnesses_T = self.calc_sg_batch(x_batch)
        f_now = softplus_batch(-dists, self.stiffness, self.kappa)

        # Apply forces to the ball.
        v_next = v_now + self.h * f_now[:,None] * normals_W
        p_next = p_now + self.h * v_next

        # Apply torques to the pivot.
        torque_now = -f_now * normals[:,0] * witnesses_T[:,1]
        omega_next = omega_now + self.h * torque_now
        theta_next = theta_now + self.h * omega_next
        return torch.hstack((p_next, theta_next[:,None], v_next, omega_next[:,None]))
    
    def rollout_batch(self, x0_batch):
        """
        Rollout system from x0 to u_trj using dynamics in batch.
        args:
        - x0 (torch.Tensor, dim: B x n): batch of initial states.
        - u_trj (torch.Tensor, dim: B x T x m): batch of input trajectories.
        returns:
        - x_trj (torch.Tensor, dim: B x (T+1) x n): batch of state trajs.
        """
        B = x0_batch.shape[0]
        x_trj_batch = torch.zeros((B, self.T+1, self.dim_x))
        x_trj_batch[:,0,:] = x0_batch

        for t in range(self.T):
            x_trj_batch[:,t+1,:] = self.dynamics_batch(
                x_trj_batch[:,t,:])

        #x_trj_batch_truncated = torch.zeros((B, self.T+1, self.dim_x))

        #for b in range(B):
        #    x_trj = x_trj_batch[b]
        #    touch_ind = torch.min(torch.argwhere(x_trj[:,1] < 0))
        #    x_trj[touch_ind:] = x_trj[touch_ind]

        return x_trj_batch    
    
    def render(self, ax, x):
        polygon = np.array([
            self.width/2 * np.array([1, 1, -1, -1]),
            self.height * np.array([1, 0, 0, 1]),
        ])
        theta = x[None,2]
        rotmat_batch = self.calc_rotation_matrix(theta).numpy()[0]
        rotated_polygon = np.einsum("ij,jk->ik", rotmat_batch, polygon)
        rotated_polygon[0,:] += self.dist 

        plt_polygon = plt.Polygon(
            np.transpose(rotated_polygon), facecolor='springgreen',
            edgecolor='black', alpha=0.1)
        plt.gca().add_patch(plt_polygon)

        plt.plot(self.dist, 0, 'ko', markersize=10)

        circle = plt.Circle(x[:2], radius=self.radius, facecolor='red', alpha=0.5, edgecolor='k')
        plt.gca().add_patch(circle)

# ds = PivotSoftPlus()
"""
plt.figure()
x_now = torch.Tensor([0.3, 0.8, 0.2, 0, 0, 0])
ds.dynamics_batch(x_now[None,:])
ds.render(plt.gca(), x_now)
dists, normals, witness, _, _ = ds.calc_sg_batch(x_now[None,:])
plt.plot(witness[0,0], witness[0,1], 'ro')
plt.axis('equal')
plt.show()

plt.figure()
xy_rotated = ds.calc_transformed_ball_coordinates(x_now[None,:])[0]
ds.dist = 0.0
ds.render(plt.gca(), torch.Tensor([xy_rotated[0], xy_rotated[1], 0.0, 0.0, 0.0, 0.0]))
plt.axis('equal')
plt.show()
"""

"""
T = 200
ds.dist = 0.5
x_trj = torch.zeros((T, 6))
x_trj[0] = torch.Tensor([0.0, 0.0, 0.0, 2.0, 2.0, 0.0])
for t in range(T - 1):
    x_trj[t+1] = ds.dynamics_batch(x_trj[None,t])[0]
    plt.figure(figsize=(8,8))
    ds.render(plt.gca(), x_trj[t])
    plt.plot(x_trj[:t,0],x_trj[:t,1], 'r-')
    plt.axis('equal')
    plt.xlim([-0.3, 1.3])
    plt.ylim([-0.6, 0.6])
    plt.savefig("animation/{:05d}.png".format(t))
    plt.close()
"""