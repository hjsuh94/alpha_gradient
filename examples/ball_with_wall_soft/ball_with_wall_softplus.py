import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import torch

from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.smoothing import (relu, relu_batch,
    softplus, softplus_batch)

class BallWithWallSoftPlus(DynamicalSystem):
    def __init__(self):
        super().__init__()
        self.h = 0.004
        self.dim_x = 4
        self.dim_u = 0

        self.d = 1
        self.v0 = 5
        self.T = 2000

        self.g = 5.0
        self.e = 0.9

        # Geometrical parameters
        self.width = 0.1 # width of the tower
        self.height = 0.5 # height of the tower
        self.dist = 1.5 # distance to the tower.

        self.kappa = np.power(10, 5)

        self.radius = 0.1
        self.center = torch.tensor([self.dist, self.height])

        self.stiffness = np.power(10, 4)

    def compute_sg(self, x):
        normal_coord = x - torch.tensor([self.dist, self.height])
        xn = normal_coord[0]
        yn = normal_coord[1]
        padwidth = self.width + 2.0 * self.radius

        if (yn < 0) and  (xn < 0):
            normal = torch.tensor([-1, 0])
            dist = -padwidth/2 - xn
        elif (yn < 0) and  (xn > 0):
            normal = torch.tensor([1, 0])
            dist = xn - padwidth/2
        elif (yn > 0) and (torch.abs(xn) < self.width/2):
            normal = torch.tensor([0, 1])
            dist = yn - self.radius
        else:
            if (yn > 0) and (xn > self.width/2):
                center = torch.tensor([self.width/2, 0])
            elif (yn > 0) and (xn < -self.width/2):
                center = torch.tensor([-self.width/2, 0])
            else:
                raise ValueError("shouldn't reach here.")
            dist_to_center = torch.linalg.vector_norm(normal_coord - center)
            dist = dist_to_center - self.radius
            normal = (normal_coord - center) / dist_to_center

        return dist, normal

    def compute_sg_batch(self, x_batch):
        #print(x_batch.shape)
        xn = x_batch[:,0] - self.dist
        yn = x_batch[:,1] - self.height
        padwidth = self.width + 2.0 * self.radius

        B = x_batch.shape[0]
        dists = torch.zeros(B)
        normals = torch.zeros(B,2)

        ynxn = torch.logical_and((yn < 0), (xn < 0))
        normals[ynxn,:] = torch.tensor([-1.0, 0.0])
        dists[ynxn] = -padwidth / 2 - xn[ynxn]

        ynxp = torch.logical_and((yn < 0), (xn > 0))
        normals[ynxp,:] = torch.tensor([1.0, 0.0])
        dists[ynxp] = xn[ynxp] - padwidth / 2

        ynxm = torch.logical_and((yn > 0), (torch.abs(xn) < self.width/2))
        normals[ynxm,:] = torch.tensor([0.0, 1.0])
        dists[ynxm] = yn[ynxm] - self.radius

        quad_1 = torch.logical_and((xn > self.width/2), (yn > 0))
        center = torch.tensor([self.dist + self.width/2, self.height])
        dist_to_center = torch.linalg.norm(
            x_batch[quad_1,:] - center, dim=1)
        dists[quad_1] = dist_to_center - self.radius
        normals[quad_1,:] = (x_batch[
            quad_1,:] - center) / dist_to_center.unsqueeze(1)

        quad_2 = torch.logical_and((xn < -self.width/2), (yn > 0))
        center = torch.tensor([self.dist -self.width/2, self.height])
        dist_to_center = torch.linalg.norm(
            x_batch[quad_2,:] - center, dim=1)
        dists[quad_2] = dist_to_center - self.radius
        normals[quad_2,:] = (x_batch[
            quad_2,:] - center) / dist_to_center.unsqueeze(1)

        return dists, normals

    def dynamics(self, x):
        p_now = x[0:2].clone()
        v_now = x[2:4].clone()

        # Compute forces. 
        dist, normal = self.compute_sg(p_now)

        f_now = softplus(-dist, self.stiffness, self.kappa) * normal
        v_next = v_now + self.h * (
            f_now + torch.tensor([0.0, -self.g]))
        p_next = p_now + self.h * v_next

        return torch.hstack((p_next, v_next))

    # Feature not complete.
    def dynamics_batch(self, x_batch):
        B = x_batch.shape[0]
        p_now = x_batch[:,0:2].clone()
        v_now = x_batch[:,2:4].clone()

        dists, normals = self.compute_sg_batch(p_now)
        forcemag_batch = softplus_batch(-dists, self.stiffness, self.kappa)
        f_now = forcemag_batch.unsqueeze(1) * normals

        v_next = v_now + self.h * (
            f_now + torch.tensor([0.0, -self.g]).reshape((1,2)))
        p_next = p_now + self.h * v_next

        return torch.hstack((p_next, v_next))

    def rollout(self, theta):
        """
        Rollout system from x0 to u_trj using dynamics.
        args:
        - x0 (torch.Tensor, dim: n): initial states.
        - u_trj (torch.Tensor, dim: T x m): input trajectories.
        returns:
        - x_trj (torch.Tensor, dim: (T+1) x n): batch of state trajs.
        """
        T = self.T
        x_trj = torch.zeros((T+1, self.dim_x))
        x_trj[0,2] = self.v0 * torch.cos(theta)
        x_trj[0,3] = self.v0 * torch.sin(theta)

        for t in range(T):
            x_trj[t+1,:] = self.dynamics(x_trj[t,:])
        return x_trj        

    def rollout_batch(self, theta_batch):
        """
        Rollout system from x0 to u_trj using dynamics in batch.
        args:
        - x0 (torch.Tensor, dim: B x n): batch of initial states.
        - u_trj (torch.Tensor, dim: B x T x m): batch of input trajectories.
        returns:
        - x_trj (torch.Tensor, dim: B x (T+1) x n): batch of state trajs.
        """
        B = theta_batch.shape[0]
        x_trj_batch = torch.zeros((B, self.T+1, self.dim_x))
        x_trj_batch[:,0,2] = self.v0 * torch.cos(theta_batch)
        x_trj_batch[:,0,3] = self.v0 * torch.sin(theta_batch)

        for t in range(self.T):
            x_trj_batch[:,t+1,:] = self.dynamics_batch(
                x_trj_batch[:,t,:])

        #x_trj_batch_truncated = torch.zeros((B, self.T+1, self.dim_x))

        #for b in range(B):
        #    x_trj = x_trj_batch[b]
        #    touch_ind = torch.min(torch.argwhere(x_trj[:,1] < 0))
        #    x_trj[touch_ind:] = x_trj[touch_ind]

        return x_trj_batch

    def render(self, ax):
        polygon = np.array([
            self.dist + self.width/2 * np.array([1, 1, -1, -1]),
            self.height * np.array([1, -1, -1, 1]),
        ])
        plt_polygon = plt.Polygon(
            np.transpose(polygon), facecolor='springgreen',
            edgecolor='black', alpha=0.1)
        plt.gca().add_patch(plt_polygon)

        polygon = np.array([
            self.dist + self.width/2 * np.array([1, 1, -1, -1]),
            self.height * np.array([1, -1, -1, 1]),
        ])

        plt_polygon = plt.Polygon(
            np.transpose(polygon), facecolor='none',
            edgecolor='black', alpha=1.0)
        plt.gca().add_patch(plt_polygon)
