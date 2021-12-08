import numpy as np

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

    def gradient(self, x, w):
        """
        Evaluate gradient at x. Noise can be added as an optional parameter.
        input: x of shape n.
        output: dfdx of shape n.
        """
        raise NotImplementedError("This method is virtual")

    def zero_order_gradient(self, x, w):
        """
        Evaluate zero-order gradient at point x.
        """
        return w * (self.evaluate(x, w) - self.evaluate(x, np.zeros(self.d)))

    def gradient_batch(self, x, w):
        """
        Evaluate gradient at x. Noise can be added as an optional parameter.
        input: x of shape n, w of shape (B, m).
        output: batch of dfdx of shape (B, n).
        """
        raise NotImplementedError("This method is virtual")

    def first_order_batch_gradient(self, x, sample_size, variance):
        """
        Compute the first order batch gradient, evaluated at x.
        """
        samples = np.random.normal(0.0, variance, (sample_size, self.d))
        batch = self.gradient_batch(x, samples)
        return np.sum(batch, axis=0) / sample_size

    def zero_order_batch_gradient(self, x, sample_size, variance):
        """
        Compute the zero order batch gradient, evaluated at x.
        """
        samples = np.random.normal(0.0, variance, (sample_size, self.d))
        batch_gradient = np.zeros(self.d)
        for k in range(sample_size):
            batch_gradient += self.zero_order_gradient(x, samples[k,:])
        batch_gradient /= sample_size
        return batch_gradient / (variance ** 2.0)

    def alpha_order_batch_gradient(self, x, sample_size, variance, alpha):
        """
        Compute the first order batch gradient, evaluated at x.
        """
        samples = np.random.normal(0.0, variance, (sample_size, self.d))

        batch = self.gradient_batch(x, samples)        
        fobg = np.average(batch, axis=0)

        batch_gradient = np.zeros(self.d)
        for k in range(sample_size):
            batch_gradient += self.zero_order_gradient(x, samples[k,:])
        batch_gradient /= sample_size
        zobg = batch_gradient / (variance ** 2.0)
        
        return alpha * fobg + (1 - alpha) * zobg
