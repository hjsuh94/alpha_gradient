import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import torch

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

class BallWithWallSoft(ObjectiveFunction):
    def __init__(self):
        super().__init__(1)
        self.d = 1
        self.v0 = 1
        self.T = 80
        self.h = 0.01
        self.g = 9.81
        self.e = 0.9

        # Geometrical parameters
        self.width = 0.1 # width of the tower
        self.height = 1.0 # height of the tower
        self.dist = 1.0 # distance to the tower.

        self.radius = self.width / 2
        self.center = torch.tensor([self.dist, self.height])

    def collide_with_bar(self, x):
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
        collision = (x[1] >= 0.0)
        return collision


    def rollout(self, theta0):

        x_trj = torch.zeros(self.T+1, 4)

        # Set initial condition.
        x_trj[0,0:2] = 0.0
        x_trj[0,2] = self.v0 * torch.cos(theta0)
        x_trj[0,3] = self.v0 * torch.cos(theta0)

        # Do semi-implicit integration
        for t in range(self.T):
            

        

    def evaluate(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        x = torch.Tensor(x)
        w = torch.Tensor(w).unsqueeze(0)

        theta = x + w

        vx = torch.cos(theta)
        vy = torch.cos(theta)
        px = torch.zeros(1)
        py = torch.zeros(1)

        # Do semi-implicit integration.
        for t in range(self.T):
            vx_next_bar = vx + self.h
            vy_next_bar = vy + self.h * (-self.g)
            px_next_bar = px + self.h * vx_next_bar
            py_next_bar = py + self.h * vy_next_bar

            p_now = torch.tensor([px, py])
            p_next_bar = torch.tensor([px_next_bar, py_next_bar])

            if (self.collision_with_ball(p_next_bar)):
                # Compute toi.



            if (self.collision_with_bar(p_next_bar)):

        


        return cost_array.detach().cpu().numpy()[0]

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)
        B = w.shape[0]

        x = torch.Tensor(x)
        w = torch.Tensor(w)

        x_final = ball_trajectory(
            self.v0, x+w, self.dball, self.hball, self.gball)
        cost_array = -torch.pow(x_final, 2.0).squeeze(1)
        return cost_array.detach().cpu().numpy()

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        x = torch.Tensor(x, requires_grad=True)
        w = torch.Tensor(w)

        x_autodiff = InitializeAutoDiff(x+w)
        x_final, trj = ball_trajectory(
            self.v0, x_autodiff, self.dball, self.hball, self.gball)
        x_final_cost = -x_final ** 2.0
        dfdx = ExtractGradient(x_final_cost)
        return dfdx

    def gradient_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        B = w.shape[0]

        z = torch.tensor(x + w, requires_grad=True, dtype=torch.float32)
        x_final = ball_trajectory(
            self.v0, z, self.dball, self.hball, self.gball)
        cost_array = -torch.sum(torch.pow(x_final, 2.0))

        cost_array.backward()
        return z.grad.detach().cpu().numpy()

"""
obj = BallWithWallTorch()
print(obj.evaluate(np.array([0.5]), np.array([0.0])))
print(obj.evaluate_batch(np.array([0.5]), np.random.normal(0, 0.1, (10,1))))
print(obj.gradient_batch(np.array([0.5]), np.random.normal(0, 0.1, (10,1))))
"""