import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import pydrake.autodiffutils
from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

def ball_trajectory(v0, theta, d, h, g, get_trajectory=False):
    xt = 2 * (v0 ** 2) * np.sin(theta) * np.cos(theta) / g
    if xt <= d:
        x_final = xt 
    else:
        ball_height = -0.5 * g / (v0 ** 2.0) * (1 / (
            np.cos(theta) ** 2.0)) * (d ** 2.0) + np.tan(theta) * d
        if ball_height <= h:
            x_final = d * (xt / xt)
        else:
            x_final = xt

    if get_trajectory:
        x_trj = np.linspace(0, x_final, 1000)
        y_trj = -0.5 * g / (v0 ** 2.0) * (1 / (
            np.cos(theta) ** 2.0)) * (x_trj ** 2.0) + np.tan(theta) * x_trj
        trj = np.vstack((x_trj, y_trj))
    else:
        trj = None

    return x_final, trj

class BallWithWall(ObjectiveFunction):
    def __init__(self):
        super().__init__(1)
        self.d = 1
        self.v0 = 1
        self.dball = 0.06
        self.hball = 0.02
        self.gball = 9.81

    def evaluate(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)
        x_final, trj = ball_trajectory(
            self.v0, x+w, self.dball, self.hball, self.gball)
        return -x_final ** 2.0

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)
        B = w.shape[0]

        cost_array = np.zeros(B)
        for i in range(B):
            x_final, trj = ball_trajectory(
                self.v0, x+w[i], self.dball, self.hball, self.gball)
            cost_array[i] = -x_final  ** 2.0
        return cost_array

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)
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
        dfdx_array = np.zeros((B,1))
        for b in range(B):
            x_autodiff = InitializeAutoDiff(x + w[b])
            x_final, trj = ball_trajectory(
                self.v0, x_autodiff, self.dball, self.hball, self.gball)
            x_final_cost = -x_final ** 2.0
            dfdx = ExtractGradient(x_final_cost)
            dfdx_array[b] = dfdx
        return dfdx_array