import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

class SoftHeaviside(ObjectiveFunction):
    def __init__(self, delta):
        """
        Multi-dimensional heaviside function. The And version is defined as
        follows:
        H(x) = 1 if all x_i satisfies x_i >= 0. 
               0 if any x_i satisfies x_i < 0.
        """
        super().__init__(1)
        self.delta = delta
        self.d = 1

    def evaluate(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        noise_array = x+w
        output = np.zeros(noise_array.shape)
        output[noise_array <= -self.delta/2] = -1.0
        output[noise_array >= self.delta/2] = 1.0
        output[
            np.logical_and((noise_array > -self.delta/2),
                (noise_array < self.delta/2)).squeeze(), :] = 2.0 * (
                    1./self.delta) * (
                noise_array[
                    np.logical_and((noise_array > -self.delta/2),
                        (noise_array < self.delta/2)).squeeze(), :])        
        
        return output

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        noise_array = x+w
        output = np.zeros(noise_array.shape)
        output[noise_array <= -self.delta/2] = -1.0
        output[noise_array >= self.delta/2] = 1.0
        output[
            np.logical_and((noise_array > -self.delta/2),
                (noise_array < self.delta/2)).squeeze(), :] = 2.0 * (
                    1./self.delta) * (
                noise_array[
                    np.logical_and((noise_array > -self.delta/2),
                        (noise_array < self.delta/2)).squeeze(), :])
        
        return np.squeeze(output)

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        dfdx = np.zeros(self.d)
        return dfdx

    def gradient_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        noise_array = x+w
        output = np.zeros(noise_array.shape)
        output[noise_array <= -self.delta/2] = 0.0
        output[noise_array >= self.delta/2] = 0.0
        output[
            np.logical_and((noise_array > -self.delta/2),
                (noise_array < self.delta/2)).squeeze(), :] = 2.0 * (
                    1./self.delta) # * (
                #noise_array[
                 #   np.logical_and((noise_array > -self.delta/2),
                 #       (noise_array < self.delta/2)).squeeze(), :])
        
        return output

