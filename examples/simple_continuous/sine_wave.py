import numpy as np
import time
import warnings
warnings.filterwarnings('error')

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

class Sine(ObjectiveFunction):
    def __init__(self, omega):
        """
        A simple sine function generator. Generates y=sin(omega  * (x + w))
        """
        super().__init__(1)

        self.omega = omega
        self.d = 1
    
    def evaluate(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)
        return np.sin(self.omega * (x + w))

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)
        return np.sin(self.omega * (x + w))

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        return self.omega * np.cos(self.omega * (x + w))

    def gradient_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)
        return self.omega * np.cos(self.omega * (x + w))        
