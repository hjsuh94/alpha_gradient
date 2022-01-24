import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import torch

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.dynamical_system import DynamicalSystem

class BallWithWallSoftDynamics(DynamicalSystem):
    def __init__(self):
        super().__init__()
        self.h = 0.005      
        self.dim_x = 4
        self.dim_u = 0

        self.d = 1
        self.v0 = 5
        self.T = 200

        self.g = 9.81
        self.e = 0.9

        # Geometrical parameters
        self.width = 0.1 # width of the tower
        self.height = 0.5 # height of the tower
        self.dist = 1.5 # distance to the tower.

        self.radius = self.width / 2
        self.center = torch.tensor([self.dist, self.height])

    def collision_with_bar(self, x):
        collision_x = (torch.abs(x[0] - self.dist) <= self.width/2)
        collision_y = (x[1] <= self.height)

        collision = torch.logical_and(collision_x, collision_y)
        return collision

    def collision_with_ball(self, x):
        collision_ball = (torch.norm(x - self.center) <= self.radius)
        collision_half = (x[1] >= self.height)

        collision = torch.logical_and(collision_ball, collision_half)
        return collision

    def collision_with_ground(self, x):
        collision = (x[1] <= -0.1)
        return collision

    def collision_with_bar_batch(self, x):
        collision_x = (torch.abs(x[:,0] - self.dist) <= self.width/2)
        collision_y = (x[:,1] <= self.height)

        collision = torch.logical_and(collision_x, collision_y)
        return collision

    def collision_with_ball_batch(self, x):
        collision_ball = (torch.norm(x - self.center, dim=1) <= self.radius)
        collision_half = (x[:,1] >= self.height)

        collision = torch.logical_and(collision_ball, collision_half)
        return collision

    def collision_with_ground_batch(self, x):
        collision = (x[:,1] <= -0.1)
        return collision        

    def dynamics(self, x):
        p_now = x[0:2].clone()
        v_now = x[2:4].clone()

        v_next_bar = v_now + self.h * torch.tensor([0.0, -self.g])
        p_next_bar = p_now + self.h * v_next_bar

        toi = 0.0
        v_next = v_next_bar

        # Collision with tower.
        if (self.collision_with_bar(
            p_next_bar) or self.collision_with_ball(p_next_bar)):

            # Collision with bar.
            if (self.collision_with_bar(p_next_bar)):
                v_next = torch.zeros(2)
                v_next[0] = -v_next_bar[0]
                v_next[1] =  v_next_bar[1]

                toi = self.h * ((self.dist -  self.width/2) - p_now) / (
                    p_next_bar - p_now)
                p_next = p_now + toi * v_next_bar + (self.h - toi) * v_next

            # Collision with ball. toi computation is more complicated.
            else:
                # 1. compute angle of the triangle.
                A = torch.norm(p_now - self.center)
                B = torch.norm(p_now - p_next_bar)
                C = torch.norm(p_next_bar - self.center)
                alpha = torch.acos(
                    (torch.pow(A,2) + torch.pow(B,2) - torch.pow(C,2)) / (
                        2.0 * A * B))
                
                # 2. Compute toi.
                X = A * torch.cos(alpha) - torch.sqrt(
                    self.radius ** 2.0 - torch.pow(A * torch.sin(alpha), 2))
                toi = self.h * X / B
                poi = p_now + v_next_bar * toi

                # 3. Compute velocity post-impact.
                theta = torch.atan((poi[1] - self.center[1]) / (
                    poi[0] - self.center[0]))
                print(theta)

                R_WP_next = torch.zeros(2,2)
                R_WP_next[0,0] = torch.cos(theta)
                R_WP_next[1,1] = torch.cos(theta)
                R_WP_next[0,1] = -torch.sin(theta)
                R_WP_next[1,0] = torch.sin(theta)

                R_PW_next = torch.zeros(2,2)
                R_PW_next[0,0] = torch.cos(theta)
                R_PW_next[1,1] = torch.cos(theta)
                R_PW_next[0,1] = torch.sin(theta)
                R_PW_next[1,0] = -torch.sin(theta)

                v_next_bar_P = torch.matmul(R_PW_next, v_next_bar)
                v_next_P = torch.zeros(2)
                v_next_P[0] = -v_next_bar_P[0]
                v_next_P[1] = v_next_bar_P[1]
                v_next = torch.matmul(R_WP_next, v_next_P)

        if (self.collision_with_ground(p_next_bar)):
            toi = self.h * (p_now + 0.1) / (torch.norm(p_next_bar - p_now))
            v_next = torch.zeros(2)
            v_next[0] = v_next_bar[0]
            v_next[1] = -v_next_bar[1]

        p_next = p_now + toi * v_next_bar + (self.h - toi) * v_next      
        return torch.hstack((p_next, v_next))

    def dynamics_batch(self, x_batch):
        B = x_batch.shape[0]
        p_now = x_batch[:,0:2].clone()
        v_now = x_batch[:,2:4].clone()

        v_next_bar = v_now + self.h * torch.tensor([0.0, -self.g])
        p_next_bar = p_now + self.h * v_next_bar

        toi = 0.0
        v_next = v_next_bar

        collision_tower = self.collision_with_bar_batch(p_next_bar).float()
        collision_ball = self.collision_with_ball_batch(p_next_bar).float()
        collision_grd = self.collision_with_ground_batch(p_next_bar).float()
        collision_towerball = collision_tower *  collision_ball

        collision_tower = collision_tower.repeat(2,1).transpose(0,1)
        collision_ball = collision_ball.repeat(2,1).transpose(0,1)
        collision_grd = collision_grd.repeat(2,1).transpose(0,1)
        collision_towerball = collision_towerball.repeat(2,1).transpose(0,1)

        print(collision_tower.shape)

        # Collision with tower.
        v_next_tower = torch.zeros(B,2)
        v_next_tower[:,0] = -v_next_bar[:,0]
        v_next_tower[:,1] =  v_next_bar[:,1]

        toi_tower = self.h * ((self.dist -  self.width/2) - p_now) / (
            p_next_bar - p_now)

        # Collision with ball.

        # 1. compute angle of the triangle.
        a = torch.norm(p_now - self.center, dim=1)
        b = torch.norm(p_now - p_next_bar, dim=1)
        c = torch.norm(p_next_bar - self.center, dim=1)

        alpha = torch.acos(
            (torch.pow(a,2) + torch.pow(b,2) - torch.pow(c,2)) / (
                2.0 * a * b))
        alpha[torch.isnan(alpha)] = 0.0

        # 2. Compute toi.
        x = a * torch.cos(alpha) - torch.sqrt(
            self.radius ** 2.0 - torch.pow(a * torch.sin(alpha), 2))
        x[torch.isnan(x)] = 0.0

        toi_ball = self.h * x / b
        toi_ball = toi_ball.repeat(2,1).transpose(0,1)
        poi = p_now + v_next_bar * toi_ball

        # 3. Compute velocity post-impact.
        theta = torch.atan((poi[:,1] - self.center[1]) / (
            poi[:,0] - self.center[0]))
        theta[torch.isnan(theta)] = 0.0

        R_WP_next = torch.zeros(B,2,2)
        R_WP_next[:,0,0] = torch.cos(theta)
        R_WP_next[:,1,1] = torch.cos(theta)
        R_WP_next[:,0,1] = -torch.sin(theta)
        R_WP_next[:,1,0] = torch.sin(theta)

        R_PW_next = torch.zeros(B,2,2)
        R_PW_next[:,0,0] = torch.cos(theta)
        R_PW_next[:,1,1] = torch.cos(theta)
        R_PW_next[:,0,1] = torch.sin(theta)
        R_PW_next[:,1,0] = -torch.sin(theta)

        v_next_bar_P = torch.bmm(R_PW_next, v_next_bar.unsqueeze(2)).squeeze(2)
        v_next_P = torch.zeros(B,2)
        v_next_P[:,0] = -v_next_bar_P[:,0]
        v_next_P[:,1] = v_next_bar_P[:,1]
        v_next_ball = torch.bmm(R_WP_next, v_next_P.unsqueeze(2)).squeeze(2)

        # Collision with ground
        toi_grd = self.h * (p_now + 0.1) / (
            torch.norm(p_next_bar - p_now, dim=1).repeat(2,1).transpose(0,1))
        v_next_grd = torch.zeros(B,2)
        v_next_grd[:,0] = v_next_bar[:,0]
        v_next_grd[:,1] = -v_next_bar[:,1]


        # v_next_tower, toi_tower, v_next_ball, toi_ball, v_next_grd, toi_grd
        toi_tower[torch.isnan(toi_tower)] = 0.0
        toi_ball[torch.isnan(toi_ball)] = 0.0
        toi_grd[torch.isnan(toi_grd)] = 0.0

        toi = collision_towerball * (
            collision_tower * toi_tower + (
                collision_ball * toi_ball + collision_grd * toi_grd))
        v_next = collision_towerball * (
            collision_tower * v_next_tower + (
                collision_ball * v_next_ball + collision_grd * v_next_grd))

        p_next = p_now + toi * v_next_bar + (self.h - toi) * v_next      

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
        return x_trj_batch

    def render(self, ax):
        polygon = np.array([
            self.dist + self.width/2 * np.array([1, 1, -1, -1]),
            self.height * np.array([1, 0, 0, 1]),
        ])
        plt_polygon = plt.Polygon(
            np.transpose(polygon), facecolor='springgreen',
            edgecolor='springgreen', alpha=0.1)
        plt.gca().add_patch(plt_polygon)

        circle = plt.Circle(
            (self.dist, self.height), self.radius, color='g', alpha=0.1)

        plt.gca().add_patch(circle)



