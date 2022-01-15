import numpy as np
from alpha_gradient.statistical_analysis import (
    compute_mean, compute_variance_norm,
    compute_confidence_interval
)

class ObjectiveFunction:
    def __init__(self, d):
        """
        Base objective function.
        """
        self.d = d # dimension of the decision variable.

    def evaluate(self, x, w):
        """
        Evaluate at x. Noise can be added as an optional parameter.
        input: x of shape n
        output: f(x) of scalar.
        """
        raise NotImplementedError("This method is virtual")

    def evaluate_batch(self, x, w):
        """
        Evaluate at x in batch of w.
        Note n CAN be different from m, 
        input: x of shape n, w of shape (B, m).
        output: f(x) of shape B.
        """
        raise NotImplementedError("This method is virtual")

    def bundled_objective(self, x, sample_size, variance):
        """
        Compute the value of the bundled objective.
        """
        samples = np.random.normal(0.0, variance, (sample_size, self.d))
        batch = self.evaluate_batch(x, samples)
        return (
            compute_mean(batch),
            compute_variance_norm(batch[:,None], 2))

    def gradient(self, x, w):
        """
        Evaluate gradient at x. Noise can be added as an optional parameter.
        input: x of shape n.
        output: dfdx of shape n.
        """
        raise NotImplementedError("This method is virtual")

    def zero_order_gradient(self, x, w, stdev):
        """
        Evaluate zero-order gradient at point x.
        """
        cost = self.evaluate(x, w) - self.evaluate(x, np.zeros(self.d))
        return w * cost / (stdev ** 2.0)

    def zero_order_gradient_batch(self, x, w, stdev):
        """
        Evaluate zero-order gradient batch.
        input: x of shape n, w of shape (B, m).
        output: dfdx^0(x) of shape (B, n).
        """
        B = w.shape[0]
        # This should be of shape B.
        cost = self.evaluate_batch(x, w) - self.evaluate(x, np.zeros(self.d))
        # This should be of shape B x m
        return np.multiply(cost[:,None], w) / (stdev ** 2.0)

    def gradient_batch(self, x, w):
        """
        Evaluate gradient at x. Noise can be added as an optional parameter.
        input: x of shape n, w of shape (B, m).
        output: batch of dfdx of shape (B, n).
        """
        raise NotImplementedError("This method is virtual")

    def fobg_given_samples(self, x, samples, stdev):
        """
        Compute first order batch gradient GIVEN the samples directly.
        Sample must be of shape (N, D) where N is sample size, D is 
        dimension of underlying data. 
        """
        batch = self.gradient_batch(x, samples)
        return compute_mean(batch), compute_variance_norm(batch, 2)

    def first_order_batch_gradient(self, x, sample_size, stdev):
        """
        Compute the first order batch gradient, evaluated at x.
        """
        samples = np.random.normal(0.0, stdev, (sample_size, self.d))
        return self.fobg_given_samples(x, samples, stdev)

    def zobg_given_samples(self, x, samples, stdev):
        """
        Compute zero order batch gradient GIVEN the samples directly.
        Sample must be of shape (N, D) where N is sample size, D is 
        dimension of underlying data. 
        """
        batch = self.zero_order_gradient_batch(x, samples, stdev)
        return compute_mean(batch), compute_variance_norm(batch, 2)

    def zero_order_batch_gradient(self, x, sample_size, stdev):
        """
        Compute the zero order batch gradient, evaluated at x.
        """
        samples = np.random.normal(0.0, stdev, (sample_size, self.d))
        return self.zobg_given_samples(x, samples, stdev)

    def aobg_given_samples(self, x, samples, stdev, alpha):

        fobg, _ = self.fobg_given_samples(x, samples, stdev)
        zobg, _ = self.zobg_given_samples(x, samples, stdev)
        return alpha * fobg + (1 - alpha) * zobg

    def alpha_order_batch_gradient(self, x, sample_size, stdev, alpha):
        """
        Compute the first order batch gradient, evaluated at x.
        """
        samples = np.random.normal(0.0, stdev, (sample_size, self.d))
        return self.aobg_given_samples(x, samples, stdev, alpha)

    def bias_constrained_aobg(self, x, sample_size, stdev, gamma,
        L=1e1, delta=0.95):
        """
        Compute alpha by solving a constrained optimization problem of 
        minimizing variance subject to bias <= gamma.
        - gamma: maximum bias tolerance.
        - L: max norm of the gradient. Used for confidence interval.
        - delta: confidence interval computation.
        """

        samples = np.random.normal(0.0, stdev, (sample_size, self.d))
        fobg, fobg_var = self.fobg_given_samples(x, samples, stdev)
        samples = np.random.normal(0.0, stdev, (sample_size, self.d))        
        zobg, zobg_var = self.zobg_given_samples(x, samples, stdev)

        # Compute confidence interval. 
        eps = compute_confidence_interval(
            zobg, zobg_var, sample_size, L, delta)[0]

        print(eps)
        # Check if zobg has less variance 
        if (eps > gamma):
            alpha = 0.0
            print("Too uncertain.")
        else:
            # Optimum of the variance minimization problem.
            alpha_bar = zobg_var / (fobg_var + zobg_var + 1e-5)
            diff = np.linalg.norm(fobg - zobg)
            print(diff)
            if (alpha_bar * diff <= gamma - eps):
                alpha = alpha_bar
                print("Within constraint. Setting to optimum.")
            else:
                alpha = (gamma - eps) / diff
                print("Out of constraint. Hitting against constraint.")
            print(alpha)

        assert(alpha >= 0)
        assert(alpha <= 1)

        return (alpha * fobg + (1 - alpha) * zobg), alpha
