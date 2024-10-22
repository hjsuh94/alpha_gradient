import numpy as np
import time 

class OptimizerParameters:
    def __init__(self):
        self.x0_initial = None
        self.verbose = True
        self.filename = ""

class Optimizer:
    def __init__(self, objective, params):
        """
        Base class for singlel-shooting trajectory optimizer.
        system: instance of DynamicalSystems class.
        params: instance of TrajoptParameters class.
        """ 
        self.objective = objective
        self.params = params

        self.x = self.params.x0_initial
        self.cost = self.objective.evaluate(self.x, np.zeros(len(self.x)))
        self.x_lst = [self.x]
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
            print("Iteration: {:04d} ".format(0) + " || " +
                "Current Cost: {0:05f} ".format(self.cost) + " || " +
                "Elapsed time: {0:05f} ".format(0.0))

        while True:
            self.x = self.local_descent(self.x)
            self.cost = self.objective.evaluate(self.x, np.zeros(len(self.x)))

            if (self.params.verbose):
                print("Iteration: {:04d} ".format(self.iter) + " || " +
                    "Current Cost: {0:05f} ".format(self.cost) + " || " +
                    "Elapsed time: {0:05f} ".format(
                        time.time() - self.start_time))

            self.x_lst.append(self.x)
            self.cost_lst.append(self.cost)

            if (self.iter > max_iterations):
                break

            # Go over to next iteration.
            self.iter += 1

        # Save
        np.save(self.params.filename + "_cost.npy", np.array(self.cost_lst))
        np.save(self.params.filename + "_params.npy", np.array(self.x_lst))

        return self.x, self.cost


class FobgdOptimizerParams(OptimizerParameters):
    def __init__(self):
        super().__init__()
        self.stdev = None
        self.sample_size = None
        self.step_size_scheduler = None

class FobgdOptimizer(Optimizer):
    def __init__(self, objective, params):
        """
        Base class for singlel-shooting trajectory optimizer.
        system: instance of DynamicalSystems class.
        params: instance of TrajoptParameters class.
        """ 
        super().__init__(objective, params)

    def local_descent(self, x):
        """
        Given the current x, compute the gradient and do descent.
        args:
        - x_trj (np.array/torch.Tensor, shape: (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: T x m)
        """
        # 1. Compute Fobgd
        fobg, _ = self.objective.first_order_batch_gradient(
            x, self.params.sample_size, self.params.stdev
        )

        step_size = self.params.step_size_scheduler.find_stepsize(
            self.objective.evaluate, fobg, x)

        # 2. Do gradient descent.
        xnew = x - step_size * fobg
        self.params.step_size_scheduler.step()

        return xnew

class ZobgdOptimizerParams(OptimizerParameters):
    def __init__(self):
        super().__init__()
        self.stdev = None
        self.sample_size = None
        self.step_size_scheduler = None

class ZobgdOptimizer(Optimizer):
    def __init__(self, objective, params):
        """
        Base class for singlel-shooting trajectory optimizer.
        system: instance of DynamicalSystems class.
        params: instance of TrajoptParameters class.
        """ 
        super().__init__(objective, params)

    def local_descent(self, x):
        """
        Given the current x, compute the gradient and do descent.
        args:
        - x_trj (np.array/torch.Tensor, shape: (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: T x m)
        """
        # 1. Compute Fobgd
        zobg, _ = self.objective.zero_order_batch_gradient(
            x, self.params.sample_size, self.params.stdev
        )
        
        step_size = self.params.step_size_scheduler.find_stepsize(
            self.objective.evaluate, zobg, x)

        # 2. Do gradient descent.
        xnew = x - step_size * zobg
        self.params.step_size_scheduler.step()
        return xnew

class AobgdOptimizerParams(OptimizerParameters):
    def __init__(self):
        super().__init__()
        self.stdev = None
        self.sample_size = None
        self.step_size = None
        self.alpha = None # TODO: make this a step?

class AobgdOptimizer(Optimizer):
    def __init__(self, objective, params):
        """
        Base class for singlel-shooting trajectory optimizer.
        system: instance of DynamicalSystems class.
        params: instance of TrajoptParameters class.
        """ 
        super().__init__(objective, params)

    def local_descent(self, x):
        """
        Given the current x, compute the gradient and do descent.
        args:
        - x_trj (np.array/torch.Tensor, shape: (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: T x m)
        """
        # 1. Compute Fobgd
        aobg = self.objective.alpha_order_batch_gradient(
            x, self.params.sample_size, self.params.stdev,
            self.params.alpha
        )

        step_size = self.params.step_size_scheduler.find_stepsize(
            self.objective.evaluate, aobg, x)        

        # 2. Do gradient descent.
        xnew = x - step_size * aobg
        self.params.step_size_scheduler.step()        
        return xnew

class BiasConstrainedOptimizerParams(OptimizerParameters):
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

class BiasConstrainedOptimizer(Optimizer):
    def __init__(self, objective, params):
        """
        Base class for singlel-shooting trajectory optimizer.
        system: instance of DynamicalSystems class.
        params: instance of TrajoptParameters class.
        """ 
        super().__init__(objective, params)

    def local_descent(self, x):
        """
        Given the current x, compute the gradient and do descent.
        args:
        - x_trj (np.array/torch.Tensor, shape: (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: T x m)
        """
        # 1. Compute Bias constrained.
        aobg, _ = self.objective.bias_constrained_aobg(
            x, self.params.sample_size, self.params.stdev,
            self.params.gamma, self.params.L, self.params.delta)

        step_size = self.params.step_size_scheduler.find_stepsize(
            self.objective.evaluate, aobg, x)        

        # 2. Do gradient descent.
        xnew = x - step_size * aobg
        self.params.step_size_scheduler.step()        
        return xnew
