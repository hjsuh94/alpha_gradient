import numpy as np
import time
import warnings
warnings.filterwarnings('error')

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

class SineQuadratic(ObjectiveFunction):
    def __init__(self, omega, mag):
        """
        A simple sine function generator. Generates y=sin(omega  * (x + w))
        """
        super().__init__(1)

        self.omega = omega
        self.mag = mag
        self.d = 1
    
    def evaluate(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)
        z = x + w
        return np.power(z, 2) + self.mag * np.sin(self.omega * z)

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)
        z = x + w        
        return np.squeeze(np.power(z, 2) + self.mag * np.sin(self.omega * z))

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        z = x + w
        return 2.0 * z + self.mag * self.omega * np.cos(self.omega * z)

    def gradient_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)
        z = x + w
        return 2.0 * z + self.mag * self.omega * np.cos(self.omega * (x + w))        
