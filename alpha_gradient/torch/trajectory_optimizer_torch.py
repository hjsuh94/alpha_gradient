import torch
import matplotlib.pyplot as plt
import time

from alpha_gradient.trajectory_optimizer import (
    TrajoptParameters, TrajectoryOptimizer)

class TrajectoryOptimizerTorch(TrajectoryOptimizer):
    def __init__(self, system, params):
        # Don't call init to parent here to deal with the case when
        # we're using the gpu.

        self.system = system
        self.params = params

        self.Q = params.Q
        self.Qd = params.Qd
        self.R = params.R
        self.x0 = params.x0
        self.xd_trj = params.xd_trj
        self.u_trj_initial = params.u_trj_initial
        self.gpu = params.gpu

        self.u_trj = self.u_trj_initial
        self.x_trj = self.system.rollout(self.params.x0, self.u_trj, self.gpu)

        self.T = self.u_trj.shape[0] # horizon.
        self.dim_x = self.system.dim_x
        self.dim_u = self.system.dim_u

        self.cost = self.evaluate_cost(self.x_trj, self.u_trj, self.gpu)

        self.x_trj_lst = [self.x_trj]
        self.u_trj_lst = [self.u_trj]
        self.cost_lst = [self.cost]

        self.start_time = time.time()
        self.iter = 1

        self.objective = lambda u_trj: self.evaluate_cost(
            self.system.rollout(self.x0, u_trj), u_trj)

    def evaluate_cost(self, x_trj, u_trj, gpu=False):
        """
        Evaluate cost given an state-input trajectory.
        - args:
            x_trj (np.array, shape (T + 1) x n): state trajectory
            u_trj (np.array, shape T x m): input trajectory
        """
        cost = 0.0

        for t in range(self.T):
            et = x_trj[t, :] - self.xd_trj[t, :]
            cost += (self.Q).mv(et).dot(et)
            cost += (self.R).mv(u_trj[t, :]).dot(u_trj[t, :])
        et = x_trj[self.T, :] - self.xd_trj[self.T, :]
        cost += (self.Qd).mv(et).dot(et)

        return cost

    def evaluate_cost_batch(self, x_trj, u_trj, gpu=False):
        """
        Evaluate cost given a batch of state-input trajectories.
        -args:
        x_trj (np.array, shape: B x (T+1) x n): state trajectory.
        u_trj (np.array, shape: B x T x m): input trajectory.
        """
        B = x_trj.shape[0]
        cost = torch.zeros(B)

        if (gpu):
            cost = cost.cuda()

        for t in range(self.T):
            et = x_trj[:, t, :] - self.xd_trj[t, :]
            cost += torch.diagonal(et.mm(self.Q).mm(et.transpose(0,1)))
            ut = u_trj[:,t,:]
            cost += torch.diagonal(ut.mm(self.R).mm(ut.transpose(0,1)))
        et = x_trj[:, self.T, :] - self.xd_trj[self.T, :]
        cost += torch.diagonal(et.mm(self.Qd).mm(et.transpose(0,1)))
        return cost


    def iterate(self, max_iterations):
        """
        Iterate local descent until convergence.
        """
        print("Iteration: {:02d} ".format(0) + " || " +
              "Current Cost: {0:05f} ".format(self.cost) + " || " +
              "Elapsed time: {0:05f} ".format(0.0))

        while True:
            x_trj_new, u_trj_new = self.local_descent(self.x_trj, self.u_trj)
            cost_new = self.evaluate_cost(x_trj_new, u_trj_new, self.gpu)

            print("Iteration: {:02d} ".format(self.iter) + " || " +
                  "Current Cost: {0:05f} ".format(cost_new) + " || " +
                  "Elapsed time: {0:05f} ".format(
                      time.time() - self.start_time))

            self.x_trj_lst.append(x_trj_new.cpu().detach().numpy())
            self.u_trj_lst.append(u_trj_new.cpu().detach().numpy())
            self.cost_lst.append(cost_new.cpu().detach().numpy())

            if (self.iter > max_iterations):
                break

            # Go over to next iteration.
            self.cost = cost_new
            self.x_trj = x_trj_new
            self.u_trj = u_trj_new
            self.iter += 1

        self.x_trj = self.x_trj.cpu().detach().numpy()
        self.u_trj = self.u_trj.cpu().detach().numpy()
        self.cost = self.cost.cpu().detach().numpy()

        return self.x_trj, self.u_trj, self.cost
