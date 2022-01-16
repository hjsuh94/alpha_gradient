import numpy as np
import time 

class PolicyOptimizerParameters:
    def __init__(self):
        self.theta0 = None
        self.verbose = True
        self.sample_size = None

class PolicyOptimizer:
    def __init__(self, objective, params):
        """
        Base class for singlel-shooting trajectory optimizer.
        system: instance of DynamicalSystems class.
        params: instance of TrajoptParameters class.
        """ 
        self.objective = objective
        self.params = params
        self.sample_size = self.params.sample_size

        self.theta = self.params.theta0
        self.cost = self.objective.evaluate_expected(
            self.theta, self.sample_size).numpy()

        print(self.cost)
        self.theta_lst = [self.theta]
        self.cost_lst = [self.cost]
        self.start_time = time.time()
        self.iter = 1

    def local_descent(self, x):
        """
        Given the current x_trj and u_trj, run forward pass on trajopt and
        do gradient descent.
        args:
        - x_trj (np.array/torch.Tensor, shape: (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: T x m)
        """
        raise NotImplementedError("This method is virtual")

    def iterate(self, max_iterations):
        """
        Iterate local descent until convergence.
        """
        if (self.params.verbose):
            print("Iteration: {:02d} ".format(0) + " || " +
                "Current Cost: {0:05f} ".format(self.cost) + " || " +
                "Elapsed time: {0:05f} ".format(0.0))

        while True:
            self.theta = self.local_descent(self.theta)
            self.cost = self.objective.evaluate_expected(
                self.theta, self.sample_size).numpy()

            if (self.params.verbose):
                print("Iteration: {:02d} ".format(self.iter) + " || " +
                    "Current Cost: {0:05f} ".format(self.cost) + " || " +
                    "Elapsed time: {0:05f} ".format(
                        time.time() - self.start_time))

            self.theta_lst.append(self.theta)
            self.cost_lst.append(self.cost)

            if (self.iter > max_iterations):
                break

            # Go over to next iteration.
            self.iter += 1

        return self.theta, self.cost


class FobgdPolicyOptimizerParams(PolicyOptimizerParameters):
    def __init__(self):
        super().__init__()
        self.stdev = None
        self.step_size_scheduler = None

class FobgdPolicyOptimizer(PolicyOptimizer):
    def __init__(self, objective, params):
        """
        Base class for singlel-shooting trajectory optimizer.
        system: instance of DynamicalSystems class.
        params: instance of TrajoptParameters class.
        """ 
        super().__init__(objective, params)

    def local_descent(self, theta):
        """
        Given the current x, compute the gradient and do descent.
        args:
        - x_trj (np.array/torch.Tensor, shape: (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: T x m)
        """
        # 1. Compute Fobgd
        fobg, _ = self.objective.first_order_batch_gradient(
            theta, self.params.sample_size, self.params.stdev
        )

        step_size = self.params.step_size_scheduler.find_stepsize(
            self.objective.evaluate, fobg, theta)

        # 2. Do gradient descent.
        theta_new = theta - step_size * fobg
        self.params.step_size_scheduler.step()

        return theta_new

class ZobgdPolicyOptimizerParams(PolicyOptimizerParameters):
    def __init__(self):
        super().__init__()
        self.stdev = None
        self.step_size_scheduler = None

class ZobgdPolicyOptimizer(PolicyOptimizer):
    def __init__(self, objective, params):
        """
        Base class for singlel-shooting trajectory optimizer.
        system: instance of DynamicalSystems class.
        params: instance of TrajoptParameters class.
        """ 
        super().__init__(objective, params)

    def local_descent(self, theta):
        zobg, _ = self.objective.zero_order_batch_gradient(
            theta, self.params.sample_size, self.params.stdev
        )
        
        step_size = self.params.step_size_scheduler.find_stepsize(
            self.objective.evaluate, zobg, theta)

        # 2. Do gradient descent.
        theta_new = theta - step_size * zobg
        self.params.step_size_scheduler.step()
        return theta_new

class BCPolicyOptimizerParams(PolicyOptimizerParameters):
    def __init__(self):
        super().__init__()
        # Smoothing parameters. 
        self.stdev = None
        self.sample_size = None

        # Confidence interval parameters
        self.delta = None
        self.L = None
        self.gamma = None

        # Gradinet descent parameters.
        self.step_size_scheduler = None

class BCPolicyOptimizer(PolicyOptimizer):
    def __init__(self, objective, params):
        """
        Base class for singlel-shooting trajectory optimizer.
        system: instance of DynamicalSystems class.
        params: instance of TrajoptParameters class.
        """ 
        super().__init__(objective, params)

    def local_descent(self, theta):
        """
        Given the current x, compute the gradient and do descent.
        args:
        - x_trj (np.array/torch.Tensor, shape: (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: T x m)
        """
        # 1. Compute Bias constrained.
        aobg, _ = self.objective.bias_constrained_aobg(
            theta, self.params.sample_size, self.params.stdev,
            self.params.gamma, self.params.L, self.params.delta)

        step_size = self.params.step_size_scheduler.find_stepsize(
            self.objective.evaluate, aobg, theta)

        # 2. Do gradient descent.
        theta_new = theta - step_size * aobg
        self.params.step_size_scheduler.step()        
        return theta_new
