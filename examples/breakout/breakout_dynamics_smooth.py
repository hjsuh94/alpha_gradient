import os, shutil, subprocess

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.smoothing import softplus, softplus_batch

class BreakoutDynamics(DynamicalSystem):
    def __init__(self):
        super().__init__()        
        self.h = 0.05
        self.dim_x = 7
        self.dim_u = 3

        # Geometry parameters.
        self.pad_width = 0.8
        self.pad_height = 0.08
        self.pad_radius = self.pad_height / 2
        self.ball_radius = 0.15

        self.padball_width = self.pad_width 
        self.padball_height = self.pad_height + 2.0 * self.ball_radius
        self.padball_radius = self.pad_radius + self.ball_radius

        self.collision_loss = 1.0
        self.stiffness = np.power(10.0, 3)
        self.damping = 0.999
        self.kappa = np.power(10.0, 1.2)

        # Render parameters
        self.x_width = 2.0
        self.y_width = 3.0

    def compute_sg(self, X_PB):
        """
        Compute distance and normal vector.
        """
        x_pb = X_PB[0]
        y_pb = X_PB[1]

        dist = None
        normal = None

        if (torch.abs(x_pb) < self.padball_width / 2):
            if y_pb > 0:
                normal = torch.tensor([0.0, 1.0])
                dist = y_pb - self.padball_height / 2
            elif y_pb < 0:
                normal = torch.tensor([0.0, -1.0])
                dist = -self.padball_height / 2 - y_pb
        elif (x_pb > self.padball_width / 2):
            center = torch.tensor([self.padball_width / 2, 0])
            dist_to_center = torch.linalg.vector_norm(X_PB - center)
            dist = dist_to_center - self.padball_radius
            normal X= (X_PB - center) / dist_to_center
        elif (x_pb < -self.padball_width / 2):
            center = torch.tensor([-self.padball_width / 2, 0])
            dist_to_center = torch.linalg.vector_norm(X_PB - center)
            dist = dist_to_center - self.padball_radius
            normal = (X_PB - center) / dist_to_center
        else:
            pass

        return dist, normal

    def compute_sg_batch(self, X_PB_batch):
            """
            Compute distance and normal vector.
            X_PB_batch is a batch of vectors (B,2)
            """
            x_pb = X_PB_batch[:,0]
            y_pb = X_PB_batch[:,1]

            B = X_PB_batch.shape[0]

            dists = torch.zeros(B)
            normals = torch.zeros(B,2)

            x_mid = (torch.abs(x_pb) < self.padball_width / 2)
            x_left = (x_pb > self.padball_width / 2)
            x_right = (x_pb < -self.padball_width / 2)

            y_pos = (y_pb > 0)
            y_neg = (y_pb < 0)

            xmyp = torch.logical_and(x_mid, y_pos)
            xmyn = torch.logical_and(x_mid, y_neg)

            # Case 1.
            dists[xmyp] = y_pb[xmyp] - self.padball_height / 2
            normals[xmyp,:] = torch.tensor([0.0, 1.0])

            # Case 2.
            dists[xmyn] = -y_pb[xmyn] - self.padball_height / 2 
            normals[xmyn,:] = -torch.tensor([0.0, 1.0])

            # Case 3.
            center = torch.tensor([self.padball_width / 2, 0])
            dist_to_center = torch.linalg.norm(
                X_PB_batch[x_left,:] - center, dim=1)
            dists[x_left] = dist_to_center - self.padball_radius
            normals[x_left,:] = (X_PB_batch[
                x_left,:] - center) / dist_to_center.unsqueeze(1)

            # Case 4.
            center = torch.tensor([-self.padball_width / 2, 0])
            dist_to_center = torch.linalg.norm(
                X_PB_batch[x_right,:] - center, dim=1)
            dists[x_right] = dist_to_center - self.padball_radius
            normals[x_right,:] = (X_PB_batch[
                x_right,:] - center) / dist_to_center.unsqueeze(1)

            return dists, normals

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

        # 1 Transform the ball position in pad body coordinates.
        R_WP_now = torch.tensor([
            [torch.cos(X_WP_now[2]), -torch.sin(X_WP_now[2])],
            [torch.sin(X_WP_now[2]), torch.cos(X_WP_now[2])]
        ])
        R_PW_now = torch.tensor([
            [torch.cos(X_WP_now[2]), torch.sin(X_WP_now[2])],
            [-torch.sin(X_WP_now[2]), torch.cos(X_WP_now[2])]
        ])

        X_PB_now = torch.matmul(
            R_PW_now, X_WB_now - X_WP_now[0:2])

        # 2. Update forces based on body frame of the pad 
        dist, normal = self.compute_sg(X_PB_now)
        F_PB_now = softplus(-dist, self.stiffness, self.kappa) * normal

        # 3. Transform forces back to original coordinates
        F_WB_now = torch.matmul(R_WP_now, F_PB_now)

        # 4. Update the simulator semi-implicitly.
        V_WB_next = self.damping * V_WB_now + self.h * F_WB_now
        X_WB_next = X_WB_now + self.h * V_WB_next
        X_WP_next = X_WP_now + self.h * u
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

        theta = X_WP_now[:,2]

        # 1. Compute Transformed Coordinates
        R_WP_now = torch.zeros(B,2,2)
        R_WP_now[:,0,0] = torch.cos(theta)
        R_WP_now[:,1,1] = torch.cos(theta)
        R_WP_now[:,0,1] = -torch.sin(theta)
        R_WP_now[:,1,0] = torch.sin(theta)

        R_PW_now = torch.zeros(B,2,2)
        R_PW_now[:,0,0] = torch.cos(theta)
        R_PW_now[:,1,1] = torch.cos(theta)
        R_PW_now[:,0,1] = torch.sin(theta)
        R_PW_now[:,1,0] = -torch.sin(theta)

        X_WB_P = X_WB_now - X_WP_now[:,0:2]
        X_PB_now = torch.bmm(R_PW_now, X_WB_P.unsqueeze(2)).squeeze(2)
        V_PB_now = torch.bmm(R_PW_now, V_WB_now.unsqueeze(2)).squeeze(2)

        # 2. Compute forces.
        dists, normals = self.compute_sg_batch(X_PB_now)
        forcemag_batch = softplus_batch(-dists, self.stiffness, self.kappa)
        F_PB_now = forcemag_batch.unsqueeze(1) * normals

        # 3. Transform forces back to original coordinates
        F_WB_now = torch.bmm(R_WP_now, F_PB_now.unsqueeze(2)).squeeze(2)

        # 4. Update the simulator semi-implicitly.
        V_WB_next = self.damping * V_WB_now + self.h * F_WB_now
        X_WB_next = X_WB_now + self.h * V_WB_next
        X_WP_next = X_WP_now + self.h * u

        return torch.hstack((X_WB_next, V_WB_next, X_WP_next))

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

        R_WP_now = np.array([
            [np.cos(x[6]), -np.sin(x[6])],
            [np.sin(x[6]), np.cos(x[6])]])

        box_poly = R_WP_now.dot(polygon) + x[4:6].numpy()[:,None]

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

        R_WP_now = np.array([
            [np.cos(x[6]), -np.sin(x[6])],
            [np.sin(x[6]), np.cos(x[6])]])

        box_poly = R_WP_now.dot(polygon) + x[4:6].numpy()[:,None]

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
        plt.figure(figsize=(8,12))        
        self.render(x_trj[-1])
        plt.plot(x_trj[0:-1,0], x_trj[0:-1,1], 'r-')
        plt.axis('equal')
        plt.xlim([-self.x_width, self.x_width])
        plt.ylim([-self.y_width, self.y_width])

        plt.plot(xg[0], xg[1], 'ro')        
        plt.show()        

    def render_traj_horizontal(self, x_trj, xg):
        print("Rendering trajectory....")
        plt.figure(figsize=(12,8))        
        self.render_horizontal(plt.gca(), x_trj[-1])
        plt.plot(x_trj[0:-1,1], x_trj[0:-1,0], 'r-')
        plt.axis('equal')
        plt.xlim([-self.y_width, self.y_width])
        plt.ylim([-self.x_width, self.x_width])

        plt.plot(xg[1], xg[0], 'ro')        
        plt.show()

    def render_traj_video(self, x_trj, xg):
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

