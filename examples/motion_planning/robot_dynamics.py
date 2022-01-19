import numpy as np
import torch

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem
from robot_motion import RobotMap

class RobotDynamics(DynamicalSystem):
    def __init__(self, map):
        super().__init__()        
        self.h = 0.1
        self.dim_x = 2
        self.dim_u = 2
        self.map = map

    def dynamics(self, x, u):
        """
        input:
            x: dim: n batched state
            u: dim: m batched input
        output:
            xnext: dim: n next state
        """
        raise NotImplementedError("This method is virtual")

    def dynamics_batch(self, x, u):
        """
        input:
            x: dim: B x n batched state
            u: dim: B x m batched input
         output:
            xnext: dim: B x n next state
        """
        raise NotImplementedError("This method is virtual")        

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
            x_trj[:,t+1,:], collision_ind = self.dynamics_batch(x_trj[:,t,:], u_trj[:,t,:])
        return x_trj, collision_ind



class RoombaDynamics(RobotDynamics):
    def __init__(self, map):
        super().__init__(map)
        self.h = 0.01
        self.dim_x = 2
        self.dim_u = 2

    def dynamics(self, x, u):
        """
        input:
            x: dim: n batched state
            u: dim: m batched input
        output:
            xnext: dim: n next state
        """
        xnext = x + self.h * u
        if self.map.in_collision(xnext):
            return x
        else:
            return xnext

    def dynamics_batch(self, x, u):
        """
        input:
            x: dim: B x n batched state
            u: dim: B x m batched input
         output:
            xnext: dim: B x n next state
        """
        B = x.shape[0]
        xnext = torch.zeros(B, self.dim_x)

        xnext = x + self.h * u

        collision_ind = self.map.in_collision_batch(
            xnext).repeat(self.dim_x,1).transpose(0,1).int().float()

        xnext_bar = collision_ind * x + (
            1.0 - collision_ind) * (xnext)
        return xnext_bar, collision_ind[:,0]

class BicycleDynamics(RobotDynamics):
    def __init__(self, map):
        super().__init__(map)
        self.h = 0.01
        self.dim_x = 5
        self.dim_u = 2

    def dynamics_np(self, x, u):
        """
        input:
            x: dim: n batched state
            u: dim: m batched input
        output:
            xnext: dim: n next state
        """
        heading = x[2]
        v = x[3]
        steer = x[4]
        dxdt = np.vstack([
            v * np.cos(heading),
            v * np.sin(heading),
            v * np.tan(steer),
            u[0],
            u[1]
        ]).transpose(0,1)

        return x + self.h * dxdt        

    def dynamics(self, x, u):
        """
        input:
            x: dim: n batched state
            u: dim: m batched input
        output:
            xnext: dim: n next state
        """
        heading = x[2].clone()
        v = x[3].clone()
        steer = x[4].clone()
        dxdt = torch.vstack([
            v * torch.cos(heading),
            v * torch.sin(heading),
            v * torch.tan(steer),
            u[0],
            u[1]
        ]).transpose(0,1)

        return x + self.h * dxdt

    def dynamics_batch(self, x, u):
        """
        input:
            x: dim: B x n batched state
            u: dim: B x m batched input
         output:
            xnext: dim: B x n next state
        """
        heading = x[:,2].clone()
        v = x[:,3].clone()
        steer = x[:,4].clone()
        dxdt = torch.vstack([
            v * torch.cos(heading),
            v * torch.sin(heading),
            v * torch.tan(steer),
            u[:,0],
            u[:,1]
        ]).transpose(0,1)
        return x + self.h * dxdt
