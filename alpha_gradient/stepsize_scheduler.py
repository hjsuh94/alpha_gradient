import numpy as np
import torch

class StepsizeScheduler:
    def __init__(self):
        self.iter = 1

    def find_stepsize(self, objective, gradient, x0):
        raise ValueError("This method is virtual.")

    def step(self):
        self.iter = self.iter + 1


class ManualScheduler(StepsizeScheduler):
    """
    Schedule step size manually as a function of iterations.
    Takes as input a function with signature: (iter, initial_step).
    Most commonly used in training deep neural networks, where a 
    human finds a suitable good learning rate for the network.
    """    
    def __init__(self, function, initial_step):
        super().__init__()
        self.function = function
        self.initial_step = initial_step

    def find_stepsize(self, objective, gradient, x0):
        stepsize = self.function(self.iter, self.initial_step)
        return stepsize


class ArmijoGoldsteinLineSearchNp(StepsizeScheduler):
    """
    Uses backtracking line search with Armijo-Goldstein condition.
    Commonly used in convex optimization for simplicity and good
    theoretical properties.
    
    However, the requirement that the next iteration must monotonically
    decrease may hurt exploration when used in non-convex settings.
    A good summary of this is "all-in for exploitation, none for exploration."
    """
    def __init__(self, alpha, beta, max_stepsize):
        super().__init__()
        self.alpha = alpha # often set to 0.5
        self.beta = beta # often set to 0.5
        self.max_stepsize = max_stepsize
        self.max_counter = 10

    def find_stepsize(self, objective, gradient, x0):
        # we'll implement steepest descent here. simple but can be slow.
        deltax = -gradient # set to be other things in second-order methods.
        stepsize = self.max_stepsize
        for counter in range(self.max_counter):
            if (objective(x0 - stepsize * gradient) < objective(
                    x0) + self.alpha * stepsize * np.sum(gradient * deltax)):
                return stepsize
            else:
                stepsize = stepsize * self.beta
        print("No valid stepsize found. Returning zero.")
        return 0.

class ArmijoGoldsteinLineSearchTorch(StepsizeScheduler):
    """
    Uses backtracking line search with Armijo-Goldstein condition.
    Commonly used in convex optimization for simplicity and good
    theoretical properties.
    
    However, the requirement that the next iteration must monotonically
    decrease may hurt exploration when used in non-convex settings.
    A good summary of this is "all-in for exploitation, none for exploration."
    """
    def __init__(self, alpha, beta, max_stepsize):
        super().__init__()
        self.alpha = alpha # often set to 0.5
        self.beta = beta # often set to 0.5
        self.max_stepsize = max_stepsize
        self.max_counter = 20

    def find_stepsize(self, objective, gradient, x0):
        # we'll implement steepest descent here. simple but can be slow.
        deltax = -gradient # set to be other things in second-order methods.
        stepsize = self.max_stepsize
        for counter in range(self.max_counter):
            if (objective(x0 - stepsize * gradient) < objective(
                    x0) + self.alpha * stepsize * torch.sum(
                        torch.mul(gradient,deltax))):
                return stepsize
            else:
                stepsize = stepsize * self.beta
        print("No valid stepsize found. Returning zero.")
        return 0.        