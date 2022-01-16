import numpy as np
import torch

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem
from alpha_gradient.policy import Policy
from alpha_gradient.statistical_analysis import (
    compute_mean, compute_variance_norm,
    compute_confidence_interval
)

class ObjectiveFunctionPolicy(ObjectiveFunction):
    def __init__(self, H,
        dynamics: DynamicalSystem, policy: Policy):
        super().__init__(policy.d)
        """
        Objective function for policy optimization. 
        Notation:
        - theta: parameters of the policy in R^d
        - x: state in R^n
        - w: noise in R^m
        - w_trj: noise in R^(T x m)
        """
        self.d = policy.d # Policy parameter dimension (flattened)
        self.n = policy.n # State dimension.
        self.m = policy.m # Policy output dimension.
        self.H = H # Horizon for the problem.
        self.dynamics = dynamics # Dynamics Class.
        self.policy = policy # Policy Class

    def rollout_policy(self, x0, w_trj, theta):
        """
        Rollout policy from initial condition x0 and noise traj w_trj
        according to policy parameters of theta.
        input: 
        - x0: torch.Tensor, shape: (n)
        - w_trj: torch.Tensor, shape: (H, m)
        - theta: torch.Tensor, shape: (d)

        output: trajectory of states.
        - x_trj: torch.Tensor, shape: (H+1, n)
        - u_trj: torch.Tensor, shape: (H, n)
        """
        x_trj = torch.zeros((self.H + 1, self.n))
        u_trj = torch.zeros((self.H, self.m))
        x_trj[0,:] = x0
        for h in range(self.H):
            u_trj[h,:] = self.policy.evaluate_policy(
                x_trj[h,:], theta) + w_trj[h,:]
            x_trj[h+1,:] = self.dynamics.evaluate(
                x_trj[h,:], u_trj[h,:])
        return x_trj, u_trj

    def rollout_policy_batch(self, x0_batch, w_trj_batch, theta):
        """
        Rollout policy from initial condition x0 and noise traj w_trj
        according to policy parameters of theta.
        input: 
        - x0: torch.Tensor, shape: (B, n)
        - w_trj: torch.Tensor, shape: (B, H, m)
        - theta: torch.Tensor, shape: (d)

        output: trajectory of states.
        - x_trj: torch.Tensor, shape: (B, H+1, n)
        - u_trj: torch.Tensor, shape: (B, H, n)        
        """
        assert(w_trj_batch.shape[2] == self.m)

        B = w_trj_batch.shape[0]
        x_trj_batch = torch.zeros((B, self.H + 1, self.n))
        u_trj_batch = torch.zeros((B, self.H, self.m))
        x_trj_batch[:,0,:] = x0_batch

        for h in range(self.H):
            u_trj_batch[:,h,:] = self.policy.evaluate_policy_batch(
                x_trj_batch[:,h,:], theta) + w_trj_batch[:,h,:]
            x_trj_batch[:,h+1,:] = self.dynamics.dynamics_batch(
                x_trj_batch[:,h,:], u_trj_batch[:,h,:])
        return x_trj_batch, u_trj_batch

    def rollout_policy_batch_theta(self, x0_batch, w_trj_batch, theta_batch):
        """
        Variant of rollout_policy_batch that takes a batch of theta.
        """
        B = w_trj_batch.shape[0]
        x_trj_batch = torch.zeros((B, self.H + 1, self.n))
        u_trj_batch = torch.zeros((B, self.H, self.m))
        x_trj_batch[:,0,:] = x0_batch

        for h in range(self.H):
            u_trj_batch[:,h,:] = self.policy.evaluate_policy_batch_theta(
                x_trj_batch[:,h,:], theta_batch) + w_trj_batch[:,h,:]
            x_trj_batch[:,h+1,:] = self.dynamics.dynamics_batch(
                x_trj_batch[:,h,:], u_trj_batch[:,h,:])
        return x_trj_batch, u_trj_batch        

    def evaluate_cost(self, x_trj, u_trj):
        """
        Evaluate the cost given a state-input trajectory.
        input:
        - x_trj: torch.Tensor, shape: (H+1, n)
        - u_trj: torch.Tensor, shape: (H, m)
        output:
        - cost: torch.Tensor, shape: (B)
        """
        raise NotImplementedError("This method is virtual.")

    def evaluate_cost_batch(self, x_trj, u_trj):
        """
        Evaluate the cost given a state-input trajectory.
        input:
        - x_trj: torch.Tensor, shape: (B, H+1, n)
        - u_trj: torch.Tensor, shape: (B, H, m)
        output:
        - cost: torch.Tensor, shape: (B)
        """
        raise NotImplementedError("This method is virtual.")

    def evaluate(self, x0, w_trj, theta):
        """
        Evaluate value of theta with a realization of noise traj w_trj.
        and the initial condition of x0
        input: 
        - x0: torch.Tensor, shape: (n)
        - w_trj: torch.Tensor, shape: (H, m)
        - theta: torch.Tensor, shape: (d)

        output: value function of the policy,
        - V(x0, w_trj, theta): torch.Tensor, shape: (1)
        """
        x_trj, u_trj = self.rollout_policy(x0, w_trj, theta)
        return self.evaluate_cost(x_trj, u_trj)

    def evaluate_batch(self, x0_batch, w_trj_batch, theta):
        """
        Evaluate value of theta with a batch of realization of noise traj
        w_trj, and batch of initial conditions x0.
        input: 
        - x0: torch.Tensor, shape: (B, n)
        - w_trj: torch.Tensor, shape: (B, H, m)
        - theta: torch.Tensor, shape: (B, d)

        output: value function of the policy,
        - V(x0, w_trj, theta): torch.Tensor, shape: (B)
        """
        x_trj_batch, u_trj_batch = self.rollout_policy_batch(
            x0_batch, w_trj_batch, theta)
        return self.evaluate_cost_batch(x_trj_batch, u_trj_batch)

    def evaluate_batch_theta(self, x0_batch, w_trj_batch, theta_batch):
        """
        Variant of evaluate_batch that takes in batch of thetas.
        """
        x_trj_batch, u_trj_batch = self.rollout_policy_batch_theta(
            x0_batch, w_trj_batch, theta_batch)
        return self.evaluate_cost_batch(x_trj_batch, u_trj_batch)        

    def evaluate_expected(self, theta, sample_size):
        """
        Evaluate expected value over zero noise in w_trj and distribution 
        in x0.
        """
        x0_batch = self.sample_x0_batch(sample_size)
        w_trj_batch = torch.zeros(sample_size, self.H, self.m)
        return torch.mean(self.evaluate_batch(x0_batch, w_trj_batch, theta))

    def gradient(self, x0, w_trj, theta):
        """
        Evaluate gradient of theta w.r.t. V(x0, w_trj, theta).
        input: identical to evaluate
        output: gradient, np.array, shape: (d)
        """
        theta_ad = theta.clone().detach().requires_grad_(True)
        cost = self.evaluate(x0, w_trj, theta_ad)
        cost.backward()
        return theta_ad.grad.detach()

    def gradient_batch(self, x0_batch, w_trj_batch, theta):
        """
        Evaluate batch of gradients w.r.t. V(x0, w_trj, theta)
        input: identical to evaluate_batch
        output: np.array, batch of dfdx of shape (B, n).
        """
        B = w_trj_batch.shape[0]
        theta_batch = theta.repeat(B,1)
        theta_ad = theta_batch.clone().detach().requires_grad_(True)
        cost = torch.sum(self.evaluate_batch_theta(
            x0_batch, w_trj_batch, theta_ad))
        cost.backward()
        return theta_ad.grad.detach()

    def zero_order_gradient(self, x0, w_trj, theta, stdev):
        """
        Evaluate zero order stochastic gradient.
        """
        w_trj_zero = torch.zeros(self.H,self.m)
        x_trj, u_trj = self.rollout_policy(x0, w_trj, theta)
        cost = self.evaluate(x0, w_trj, theta) - self.evaluate(
            x0, w_trj_zero, theta)

        policy_likelihood = torch.zeros(self.d)
        for h in range(self.H):
            policy_likelihood = policy_likelihood + torch.matmul(
                self.policy.policy_jacboian(
                    x_trj[h,:], theta).transpose(0,1),
                w_trj[h,:])

        return cost * policy_likelihood / (stdev ** 2.0)

    def zero_order_gradient_batch(self, x0_batch, w_trj_batch, theta, stdev):
        """
        Evaluate batch of zero order stochastic gradient.
        input:
        - x0_batch: torch.Tensor of shape (B, n)
        - w_trj_batch: torch.Tensor of shape (B, H, m)
        - theta: torch.Tensor of shape (d)
        - stdev: scalar that represents injected noise on output.
        output: 
        - gradient: torch.Tensor of shape (B, d)
        """
        B = w_trj_batch.shape[0]
        w_trj_zero = torch.zeros(B,self.H,self.m)
        x_trj_batch, u_trj_batch = self.rollout_policy_batch(
            x0_batch, w_trj_batch, theta)

        cost = (self.evaluate_batch(x0_batch, w_trj_batch, theta)
         - self.evaluate_batch(x0_batch, w_trj_zero, theta)) # (B)

        policy_likelihood = torch.zeros(B, self.d, 1)
        for h in range(self.H):
            policy_likelihood = policy_likelihood + torch.bmm(
                self.policy.policy_jacobian_batch( # (B, d, m)
                    x_trj_batch[:,h,:], theta).transpose(1,2),
                w_trj_batch[:,h,:].unsqueeze(2)) # (B, m, 1)

        cost_tensor = cost.unsqueeze(1).unsqueeze(2) # (B, 1, 1)
        zog = torch.bmm(policy_likelihood, cost_tensor).squeeze(2) # (B, d)

        return zog / (stdev ** 2.0)   

    def sample_x0_batch(self, sample_size):
        """
        This method is responsible to providing an adequate distribution over
        intiail conditions that the problem needs to care about.
        input: None
        output: 
        - x0_batch: torch.tensor, shape: (B, n)
        """
        raise NotImplementedError("This method is virtual")

    def fobg_given_samples(self, x0_batch, w_trj_batch, theta, stdev):
        """
        Compute first order batch gradient GIVEN the samples directly.
        """
        batch = self.gradient_batch(x0_batch, w_trj_batch, theta).numpy()
        return compute_mean(batch), compute_variance_norm(batch, 2)

    def first_order_batch_gradient(self, theta, sample_size, stdev):
        """
        Compute the first order batch gradient, evaluated at x.
        input:
        - theta: parameter (d).
        - sample_size : sample size.
        - stdev: standard deviation of the injected noise.
        output:
        - fobg: gradient (d).

        NOTE: The reason why the sampling of x0_batch doesn't happen within
        the method is that the distribution of x0_batch can be a bit 
        application-dependent. So the user needs to sample and provide the
        distribution of x0. We provide an example function, but this method
        is meant to be overwritten.
        """
        x0_batch = self.sample_x0_batch(sample_size)
        w_trj_batch = torch.normal(
            0.0, stdev, (sample_size, self.H, self.m))
        fobg = self.fobg_given_samples(x0_batch, w_trj_batch, theta, stdev)
        return fobg

    def zobg_given_samples(self, x0_batch, w_trj_batch, theta, stdev):
        """
        Compute zero order batch gradient GIVEN the samples directly.
        """
        batch = self.zero_order_gradient_batch(
            x0_batch, w_trj_batch, theta, stdev).numpy()
        return compute_mean(batch), compute_variance_norm(batch, 2)

    def zero_order_batch_gradient(self, theta, sample_size, stdev):
        """
        Compute the zero order batch gradient, evaluated at x.
        """
        x0_batch = self.sample_x0_batch(sample_size)
        w_trj_batch = torch.normal(
            0.0, stdev, (sample_size, self.H, self.m))
        zobg = self.zobg_given_samples(x0_batch, w_trj_batch, theta, stdev)
        return zobg

    def aobg_given_samples(self, x0_batch, w_trj_batch, theta, stdev, alpha):
        fobg, _ = self.fobg_given_samples(x0_batch, w_trj_batch, theta, stdev)
        zobg, _ = self.zobg_given_samples(x0_batch, w_trj_batch, theta, stdev)
        return alpha * fobg + (1 - alpha) * zobg

    def alpha_order_batch_gradient(self, theta, sample_size, stdev, alpha):
        """
        Compute the first order batch gradient, evaluated at x.
        """
        x0_batch = self.sample_x0_batch(sample_size)
        w_trj_batch = np.random.normal(
            0.0, stdev, (sample_size, self.H, self.d))

        aobg = self.aobg_given_samples(
            x0_batch, w_trj_batch, theta, stdev, alpha)

        return aobg

    def bias_constrained_aobg(self, theta, sample_size, stdev, gamma,
        L=1e1, delta=0.95):
        """
        Compute alpha by solving a constrained optimization problem of 
        minimizing variance subject to bias <= gamma.
        - gamma: maximum bias tolerance.
        - L: max norm of the gradient. Used for confidence interval.
        - delta: confidence interval computation.
        """

        x0_batch = self.sample_x0_batch(sample_size)
        w_trj_batch = np.random.normal(0.0, stdev, (sample_size, self.d))
        fobg, fobg_var = self.fobg_given_samples(
            x0_batch, w_trj_batch, theta, stdev)

        x0_batch = self.sample_x0_batch(sample_size)
        w_trj_batch = np.random.normal(0.0, stdev, (sample_size, self.d))
        zobg, zobg_var = self.zobg_given_samples(
            x0_batch, w_trj_batch, theta, stdev)

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

        aobg = (alpha * fobg + (1 - alpha) * zobg), alpha
        return aobg

