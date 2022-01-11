import numpy as np

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

class HeavisideAllPositive(ObjectiveFunction):
    def __init__(self, d):
        """
        Multi-dimensional heaviside function. The And version is defined as
        follows:
        H(x) = 1 if all x_i satisfies x_i >= 0. 
               0 if any x_i satisfies x_i < 0.
        """
        super().__init__(d)
        self.d = d

    def evaluate(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)
        return np.all(x + w>=0).astype(np.float)

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        return np.all(x + w >= 0, axis=1).astype(np.float)

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        dfdx = np.zeros(self.d)
        return dfdx

    def gradient_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        B = w.shape[0]
        return np.zeros((B, self.d))

class HeavisideAnyPositive(ObjectiveFunction):
    def __init__(self, d):
        """
        Multi-dimensional heaviside function. The And version is defined as
        follows:
        H(x) = 1 if any x_i satisfies x_i >= 0. 
               0 if all x_i satisfies x_i < 0.
        """
        super().__init__(d)
        self.d = d

    def evaluate(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)
        return np.any(x + w>=0).astype(np.float)

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        return np.any(x + w >= 0, axis=1).astype(np.float)

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        dfdx = np.zeros(self.d)
        return dfdx

    def gradient_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        B = w.shape[0]
        return np.zeros((B, self.d))        


def test_heaviside_all():
    d = 5
    
    heaviside_all = HeavisideAllPositive(d)
    test_array = np.array([0, 0, 0, 0, 0])
    val = heaviside_all.evaluate(test_array, np.zeros(5))
    print(test_array, val)
    test_array = np.array([0, 1, 0, 3, 0])
    val = heaviside_all.evaluate(test_array, np.zeros(5))
    print(test_array, val)
    test_array = np.array([0, 0, -1, 3, 0])
    val = heaviside_all.evaluate(test_array, np.zeros(5))
    print(test_array, val)

    B = 100
    test_array = np.array([0, 0, 0, 0, 0])    
    noise_array = np.zeros((B, d))
    val = heaviside_all.evaluate_batch(test_array, noise_array)
    print(val)

    val = heaviside_all.gradient(test_array, np.zeros(5))
    print(val)

    val = heaviside_all.gradient_batch(test_array, noise_array)
    print(val)
