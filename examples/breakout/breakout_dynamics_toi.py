import os, shutil, subprocess

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem

class BreakoutDynamics(DynamicalSystem):
    def __init__(self):
        super().__init__()        
        self.h = 0.05
        self.dim_x = 7
        self.dim_u = 3

        # Geometry parameters.
        self.pad_width = 0.8
        self.pad_height = 0.05
        self.ball_radius = 0.15
        self.collision_loss = 1.0
        self.damping = 0.999
        # Render parameters
        self.x_width = 2.0
        self.y_width = 3.0

    def dynamics(self, x, u):
        """
        input:
            x: dim: n batched state
            u: dim: m batched input
        output:
            xnext: dim: n next state
        """
        X_WB_now = x[0:2].clone() # x y of ball
        V_WB_now = x[2:4].clone() # vx vy of ball
        X_WP_now = x[4:7].clone() # x y theta of pad

        # 1. Compute the next state of the simulator.
        X_WB_next = X_WB_now + self.h * V_WB_now
        X_WP_next = X_WP_now + self.h * u

        # 2. Check if the ball and pad are colliding.
        # 2.1 Transform the ball position in pad body coordinates.
        R_WP_next = torch.tensor([
            [torch.cos(X_WP_next[2]), -torch.sin(X_WP_next[2])],
            [torch.sin(X_WP_next[2]), torch.cos(X_WP_next[2])]
        ])
        R_PW_next = torch.tensor([
            [torch.cos(X_WP_next[2]), torch.sin(X_WP_next[2])],
            [-torch.sin(X_WP_next[2]), torch.cos(X_WP_next[2])]
        ])

        X_PB_next = torch.matmul(
            R_PW_next, X_WB_next - X_WP_now[0:2])

        V_PB_now = torch.matmul(R_PW_next, V_WB_now)

        V_PB_next = torch.zeros(2)
        V_PB_next[0] = self.damping * V_PB_now[0]

        # 2.2 Check for collision in transformed coordinates.
        if (torch.abs(X_PB_next[0]) <= self.pad_width + self.ball_radius) and (
            torch.abs(X_PB_next[1]) <= self.pad_height + self.ball_radius):



            V_PB_next[1] = -self.collision_loss * V_PB_now[1]
        else:
            V_PB_next[1] = self.damping * V_PB_now[1]

        # 2.3 Transform back to original coordinates.
        V_WB_next = torch.matmul(R_WP_next, V_PB_next)
        X_WB_next = X_WB_now + self.h * V_WB_next

        # 4. Update the simulator.
        return torch.hstack((X_WB_next, V_WB_next, X_WP_next))

    def dynamics_batch(self, x, u):
        """
        input:
            x: dim: B x n batched state
            u: dim: B x m batched input
         output:
            xnext: dim: B x n next state
        """
        B = x.shape[0]

        X_WB_now = x[:,0:2].clone() # x y of ball
        V_WB_now = x[:,2:4].clone() # vx vy of ball
        X_WP_now = x[:,4:7].clone() # x y theta of pad

        # 1. Compute the next state of the simulator.
        X_WB_next = X_WB_now + self.h * V_WB_now
        X_WP_next = X_WP_now + self.h * u

        theta = X_WP_next[:,2]

        # 2. Check if the ball and pad are colliding.
        # 2.1 Transform the ball position in pad body coordinates.
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

        X_WB_P = X_WB_next - X_WP_now[:,0:2]
        X_PB_next = torch.bmm(R_PW_next, X_WB_P.unsqueeze(2)).squeeze(2)
        V_PB_now = torch.bmm(R_PW_next, V_WB_now.unsqueeze(2)).squeeze(2)

        V_PB_next = torch.zeros(B,2)
        V_PB_next[:,0] = self.damping * V_PB_now[:,0]

        # 2.2 Check for collision in transformed coordinates.

        col_x = (torch.abs(X_PB_next[:,0]) <= self.pad_width + self.ball_radius)
        col_y = (torch.abs(X_PB_next[:,1]) <= self.pad_height + self.ball_radius)

        col_ind = torch.logical_and(col_x, col_y).float()

        # compute time to impact.
        X_PB_now = X_PB_next - self.h * V_PB_now
        toi = self.h * (col_ind * (
            1 - (self.pad_height + self.ball_radius - X_PB_next[:,1]) / (
            X_PB_now[:,1] - X_PB_next[:,1] + 1e-5))).repeat(2,1).transpose(0,1)
        # B vector.

        V_PB_next[:,1] = col_ind * (-self.collision_loss * V_PB_now[:,1]) + (
            1.0 - col_ind) * (self.damping * V_PB_now[:,1])

        X_WP_next_bar = col_ind.repeat(3,1).transpose(0,1) * X_WP_now + (
            1.0 - col_ind).repeat(3,1).transpose(0,1) * X_WP_next

        # 2.3 Transform back to original coordinates.
        V_WB_next_bar = torch.bmm(R_WP_next, V_PB_next.unsqueeze(2)).squeeze(2)
        X_WB_next_bar = X_WB_now + toi * V_WB_now + (self.h - toi) * V_WB_next_bar

        # 4. Update the simulator.
        return torch.hstack((X_WB_next_bar, V_WB_next_bar, X_WP_next_bar))

    def rollout(self, x0, u_trj):
        """
        Rollout system from x0 to u_trj using dynamics.
        args:
        - x0 (torch.Tensor, dim: n): initial states.
        - u_trj (torch.Tensor, dim: T x m): input trajectories.
        returns:
        - x_trj (torch.Tensor, dim: (T+1) x n): batch of state trajs.
        """
        T = u_trj.shape[0]
        x_trj = torch.zeros((T+1, self.dim_x))
        x_trj[0,:] = x0
        for t in range(T):
            x_trj[t+1,:] = self.dynamics(x_trj[t,:], u_trj[t,:])
        return x_trj

    def rollout_batch(self, x0, u_trj):
        """
        Rollout system from x0 to u_trj using dynamics in batch.
        args:
        - x0 (torch.Tensor, dim: B x n): batch of initial states.
        - u_trj (torch.Tensor, dim: B x T x m): batch of input trajectories.
        returns:
        - x_trj (torch.Tensor, dim: B x (T+1) x n): batch of state trajs.
        """
        B = u_trj.shape[0]
        T = u_trj.shape[1]
        x_trj = torch.zeros((B, T+1, self.dim_x))
        x_trj[:,0,:] = x0
        for t in range(T):
            x_trj[:,t+1,:] = self.dynamics_batch(x_trj[:,t,:], u_trj[:,t,:])
        return x_trj

    def render(self, x):
        # Draw ball
        circle = plt.Circle((x[0], x[1]),
        self.ball_radius, facecolor='r', edgecolor='black', alpha=0.6)
        plt.gca().add_patch(circle)

        # Draw pad
        polygon = np.array([
            self.pad_width / 2 * np.array([1, 1, -1, -1]),
            self.pad_height / 2 * np.array([1, -1, -1, 1]),
        ])

        R_WP_next = np.array([
            [np.cos(x[6]), -np.sin(x[6])],
            [np.sin(x[6]), np.cos(x[6])]])

        box_poly = R_WP_next.dot(polygon) + x[4:6].numpy()[:,None]

        plt_polygon = plt.Polygon(
            np.transpose(box_poly), facecolor='blue',
            edgecolor='black', alpha=0.8)
        plt.gca().add_patch(plt_polygon)

        # Draw 
        polygon = np.array([
            self.x_width * np.array([1, 1, -1, -1]),
            self.y_width * np.array([1, -1, -1, 1]),
        ])
        plt_polygon = plt.Polygon(
            np.transpose(polygon), facecolor='springgreen',
            edgecolor='springgreen', alpha=0.1)
        plt.gca().add_patch(plt_polygon)

    def render_horizontal(self, ax, x):
        # Draw ball
        circle = plt.Circle((x[1], x[0]),
        self.ball_radius, facecolor='r', edgecolor='black', alpha=0.6)
        ax.add_patch(circle)

        # Draw pad
        polygon = np.array([
            self.pad_height / 2 * np.array([1, 1, -1, -1]),
            self.pad_width / 2 * np.array([1, -1, -1, 1]),
        ])

        x[6] += np.pi 

        R_WP_next = np.array([
            [np.cos(x[6]), -np.sin(x[6])],
            [np.sin(x[6]), np.cos(x[6])]])

        box_poly = R_WP_next.dot(polygon) + x[4:6].numpy()[:,None]

        plt_polygon = plt.Polygon(
            np.transpose(box_poly), facecolor='blue',
            edgecolor='black', alpha=0.8)
        ax.add_patch(plt_polygon)

        # Draw 
        polygon = np.array([
            self.y_width * np.array([1, 1, -1, -1]),
            self.x_width * np.array([1, -1, -1, 1]),
        ])
        plt_polygon = plt.Polygon(
            np.transpose(polygon), facecolor='springgreen',
            edgecolor='springgreen', alpha=0.1)
        ax.add_patch(plt_polygon)        

    def render_traj(self, x_trj, xg):
        print("Rendering trajectory....")
        os.mkdir("temp")        
        T = x_trj.shape[0]
        for t in tqdm(range(T)):
            plt.figure(figsize=(8,12))
            self.render(x_trj[t])
            plt.plot(x_trj[0:t,0], x_trj[0:t,1], 'r-')

            plt.axis('equal')
            plt.xlim([-self.x_width, self.x_width])
            plt.ylim([-self.y_width, self.y_width])

            plt.plot(xg[0], xg[1], 'ro')

            plt.savefig("temp/{:05d}.png".format(t))
            plt.close()
    
        # ffmpeg
        subprocess.call([
            'ffmpeg', '-i', "temp/%05d.png", 'output.mp4',
            '-vcodec', 'libx264',# '-pix_fmt', 'yuv420p',
            '-crf', '18', '-preset', 'slow', '-r', '30'])

        shutil.rmtree("temp")
        print("Done!")

    def render_traj_batch(self, x_trj_batch, xg):
        print("Rendering trajectory....")
        os.mkdir("temp")        
        B = x_trj_batch.shape[0]
        T = x_trj_batch.shape[1]

        self.counter = 0

        for b in tqdm(range(B)):
            for t in range(T):
                plt.figure(figsize=(8,12))
                self.render(x_trj_batch[b,t])
                plt.plot(x_trj_batch[b,0:t,0], x_trj_batch[b,0:t,1], 'r-')

                plt.axis('equal')
                plt.xlim([-self.x_width, self.x_width])
                plt.ylim([-self.y_width, self.y_width])

                plt.plot(xg[0], xg[1], 'o', color='purple', alpha=0.8,
                    markersize=10.0)

                plt.savefig("temp/{:05d}.png".format(self.counter))
                plt.close()

                self.counter += 1
    
        # ffmpeg
        subprocess.call([
            'ffmpeg', '-i', "temp/%05d.png", 'output.mp4',
            '-vcodec', 'libx264',# '-pix_fmt', 'yuv420p',
            '-crf', '18', '-preset', 'slow', '-r', '60'])

        shutil.rmtree("temp")
        print("Done!")

