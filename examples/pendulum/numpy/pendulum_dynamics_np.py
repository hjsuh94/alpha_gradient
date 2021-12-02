import numpy as np
import pydrake.symbolic as ps
import time

from alpha_gradient.numpy.dynamical_system_np import DynamicalSystemNp

class PendulumDynamicsNp(DynamicalSystemNp):
    def __init__(self, h):
        super().__init__()
        """
        x = [q, qdot]
        u = [tau]
        """
        self.h = h
        self.dim_x = 2
        self.dim_u = 1

    def dynamics(self, x, u):
        """
        Numeric expression for dynamics.
        x (np.array, dim: n): state
        u (np.array, dim: m): action
        """

        angle = x[0]
        speed = x[1]

        # Do semi-implicit integration.
        next_speed = speed + self.h * (-np.sin(angle) + u[0])
        next_angle = angle + self.h * next_speed

        x_new = np.array([next_angle, next_speed])
        return x_new

    def dynamics_batch(self, x, u):
        """
        Batch dynamics. Uses pytorch for 
        -args:
            x (np.array, dim: B x n): batched state
            u (np.array, dim: B x m): batched input
        -returns:
            xnext (np.array, dim: B x n): batched next state
        """

        angle = x[:,0]
        speed = x[:,1]
        torque = u[:,0]

        # Do semi-implicit integration.
        next_speed = speed + self.h * (-np.sin(angle) + torque)
        next_angle = angle + self.h * next_speed

        x_new = np.vstack((next_angle, next_speed)).transpose()
        return x_new
