import numpy as np
import time
import warnings
import torch
warnings.filterwarnings('error')

from pydrake.all import InitializeAutoDiff, ExtractGradient
from alpha_gradient.objective_function import ObjectiveFunction

class LpNorm(ObjectiveFunction):
    def __init__(self, p, d):
        """
        A simple linear function generator. Generates y=|x|_p, where x\in R^d
        """
        super().__init__(d)

        self.p = p
        self.d = d
    
    def evaluate(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)
        return np.linalg.norm(x + w, self.p)

    def evaluate_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        return np.linalg.norm(x + w, self.p, axis=1)

    def gradient(self, x, w):
        assert(len(x) == self.d)
        assert(len(w) == self.d)

        dfdx = np.zeros(self.d)
        z = x + w
        z_ad = InitializeAutoDiff(z)

        norm = np.power(np.sum(np.power(z_ad, self.p)), 1./self.p)
        return ExtractGradient(np.array([norm])).squeeze(0)

    def gradient_batch(self, x, w):
        assert(len(x) == self.d)
        assert(w.shape[1] == self.d)

        B = w.shape[0]

        z = torch.tensor(x + w, requires_grad=True)
        cost_array = torch.sum(torch.norm(z, p=2, dim=1))

        cost_array.backward()
        return z.grad.detach().cpu().numpy()

def test_Lp_norm():
    p = 2
    d = 5

    lp_norm = LpNorm(p, d)
    val = lp_norm.evaluate(np.array([0, 0, 3, 0, 1]), np.zeros(5))
    print(val)

    x = np.array([-0.1, 3, 2, 5, 1])
    w = np.zeros(5)

    val = lp_norm.gradient(x, w)
    print(val)

    w = np.random.normal(0.0, 0.1, (2,5))
    val = lp_norm.gradient_batch(x, w)
    print(val)

    val = lp_norm.first_order_batch_gradient(x, 100, 0.01)
    print(val)

    val = lp_norm.zero_order_batch_gradient(x, 10000, 0.1)
    print(val)

    return None
