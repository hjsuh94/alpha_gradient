import numpy as np
import time 

class TrajoptParameters:
    def __init__(self):
        self.Q = None
        self.Qd = None
        self.R = None
        self.x0 = None
        self.xd_trj = None
        self.u_trj_initial = None

class TrajectoryOptimizer:
    def __init__(self, system, params):
        """
        Base class for singlel-shooting trajectory optimizer.
        system: instance of DynamicalSystems class.
        params: instance of TrajoptParameters class.
        """ 
        self.system = system
        self.params = params

        self.Q = params.Q
        self.Qd = params.Qd
        self.R = params.R
        self.x0 = params.x0
        self.xd_trj = params.xd_trj
        self.u_trj_initial = params.u_trj_initial

        self.u_trj = self.u_trj_initial
        self.x_trj = self.system.rollout(self.params.x0, self.u_trj)

        self.T = self.u_trj.shape[0] # horizon.
        self.dim_x = self.system.dim_x
        self.dim_u = self.system.dim_u

        self.cost = self.evaluate_cost(self.x_trj, self.u_trj)

        self.x_trj_lst = [self.x_trj]
        self.u_trj_lst = [self.u_trj]
        self.cost_lst = [self.cost]

        self.start_time = time.time()
        self.iter = 1


    def evaluate_cost(self, x_trj, u_trj):
        """
        Evaluate the value function of trajopt given x_trj, u_trj.
        args:
        - x_trj (np.array/torch.Tensor, shape: (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: T x m)
        """
        raise NotImplementedError("This method is virtual")

    def evaluate_cost_batch(self, x_trj, u_trj):
        """
        Evaluate the value function of trajopt given x_trj, u_trj in batch.
        args:
        - x_trj (np.array/torch.Tensor, shape: B x (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: B x T x m)
        """        
        raise NotImplementedError("This method is virtual")

    def local_descent(self, x_trj, u_trj):
        """
        Given the current x_trj and u_trj, run forward pass on trajopt and
        do gradient descent.
        args:
        - x_trj (np.array/torch.Tensor, shape: (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: T x m)
        """
        raise NotImplementedError("This method is virtual")

    def local_descent_batch(self, x_trj, u_trj):
        """
        Given batch of x_trj and u_trj, run forward pass on trajopt and
        do gradient descent.
        args:
        - x_trj (np.array/torch.Tensor, shape: B x (T+1) x n)
        - u_trj (np.array/torch.Tensor, shape: B x T x m)
        """
        raise NotImplementedError("This method is virtual")

    def iterate(self, max_iterations):
        """
        Iterate local descent until convergence.
        """
        print("Iteration: {:02d} ".format(0) + " || " +
              "Current Cost: {0:05f} ".format(self.cost) + " || " +
              "Elapsed time: {0:05f} ".format(0.0))

        while True:
            x_trj_new, u_trj_new = self.local_descent(self.x_trj, self.u_trj)
            cost_new = self.evaluate_cost(x_trj_new, u_trj_new)

            print("Iteration: {:02d} ".format(self.iter) + " || " +
                  "Current Cost: {0:05f} ".format(cost_new) + " || " +
                  "Elapsed time: {0:05f} ".format(
                      time.time() - self.start_time))

            self.x_trj_lst.append(x_trj_new)
            self.u_trj_lst.append(u_trj_new)
            self.cost_lst.append(cost_new)

            if (self.iter > max_iterations):
                break

            # Go over to next iteration.
            self.cost = cost_new
            self.x_trj = x_trj_new
            self.u_trj = u_trj_new
            self.iter += 1

        return self.x_trj, self.u_trj, self.cost                
