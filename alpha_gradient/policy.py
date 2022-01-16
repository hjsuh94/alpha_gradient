import numpy as np
import torch
import torch.nn as nn

class Policy():
    def __init__(self):
        """
        Parametrized policy class for policy abstraction.
        Notation:
        - theta: parameters of the policy in R^d.
        - x: state in R^n
        - w: noise in R^m
        """

    def evaluate_policy(self, x, theta):
        """
        Given state x and policy parameters theta,
        produce an output of the policy y.
        input:
        - x : torch.Tensor, shape: (n)
        - theta: torch.Tensor, shape: (d)        
        output:
        - y : torch.Tensor, shape: (m)
        """
        raise NotImplementedError("This method is virtual")

    def evaluate_policy_batch(self, x, theta):
        """
        Given batch of states x and policy parameters theta,
        produce batch of output of the policy y.
        input:
        - x : torch.Tensor, shape: (B, n)
        - theta: torch.Tensor, shape: (d)
        output:
        - y : torch.Tensor, sahpe: (B, m)
        """        
        raise NotImplementedError("This method is virtual")        

    def policy_jacobian(self, x, theta):
        """
        Given state x and policy parameters theta,
        produce an Jacobian dy/dtheta evaluated at x.
        input:
        - x : torch.Tensor, shape: (n)
        - theta: torch.Tensor, shape: (d)        
        output:
        - J : torch.Tensor, shape: (m, d)
        """        
        raise NotImplementedError("This method is virtual")

    def policy_jacobian_batch(self, x, theta):
        """
        Given batch of states x and policy parameters theta,
        produce batch of Jacobian dy/dtheta evaluated at x.
        input:
        - x : torch.Tensor, shape: (B, n)
        - theta: torch.Tensor, shape: (d)        
        output:
        - J : torch.Tensor, shape: (B, m, d)
        """
        raise NotImplementedError("This method is virtual")


# Linear Policy class.
class LinearPolicy(Policy):
    def __init__(self, n, m):
        """
        Linear policy class with a linear gain.
        Notation:
        - theta: parameters of the policy in R^d.
        - x: state in R^n
        - w: noise in R^m

        For linear policies, d = n * m (# of entries in gain matrix.)
        """
        super().__init__()
        # Compute dimension of the policy.
        self.n = n
        self.m = m
        self.d = self.n * self.m

    def evaluate_policy(self, x, theta):
        assert(len(theta) == self.d)
        theta_matrix = theta.reshape(self.m, self.n)
        return torch.matmul(theta_matrix, x)

    def evaluate_policy_batch(self, x_batch, theta):
        """
        input:
        - theta: shape (d)
        - x_batch: shape (B, n)
        output:
        - y : shape (B, m)
        """
        assert(x_batch.shape[1] == self.n)
        assert(len(theta) == self.d)

        B = x_batch.shape[0]
        theta_matrix = theta.reshape(self.m, self.n).transpose(0,1)
        y_batch = torch.matmul(x_batch.clone(), theta_matrix) # B x m
        return y_batch

    def evaluate_policy_batch_theta(self, x_batch, theta_batch):
        """
        input:
        - theta_batch: shape (B, d)
        - x_batch: shape (B, n)
        output:
        - y : shape (B, m)
        """
        B = x_batch.shape[0]
        # B x m x n
        theta_matrix = theta_batch.reshape(B, self.m, self.n)
        x_batch_tensor = x_batch.clone().unsqueeze(2) # B x n x 1

        y_batch = torch.bmm(theta_matrix, x_batch_tensor).squeeze(2)
        return y_batch        

    def policy_jacobian(self, x, theta):
        eval = lambda theta_ad: self.evaluate_policy(x, theta_ad)
        return torch.autograd.functional.jacobian(eval, theta)

    def policy_jacobian_batch(self, x_batch, theta):
        eval = lambda theta_ad: self.evaluate_policy_batch(x_batch, theta_ad)
        return torch.autograd.functional.jacobian(eval, theta)

def test_linear_policy():
    policy = LinearPolicy(5,3)
    print(policy.evaluate_policy(torch.rand(5), torch.rand(15)).shape)
    print(policy.evaluate_policy_batch(torch.rand(10, 5), torch.rand(15)).shape)

    print(policy.policy_jacobian(torch.ones(5), torch.ones(15)))
    print(policy.policy_jacobian_batch(torch.ones(10, 5), torch.ones(15)).shape)

class PolicyNLP(nn.Module):
    def __init__(self, n, m):
        super(PolicyNLP, self).__init__()

        self.policy_mlp = nn.Sequential(
            nn.Linear(n, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, m)
        )

    def forward(self, x):
        return self.policy_mlp(x)

    def state_dict_to_tensor(self, state_dict):
        return None

    def param_tensor_to_state_dict(self, state_dict):
        return None

"""
Below is WIP.
"""

# Neural Policy class.
class NNPolicy(Policy):
    # WIP.
    def __init__(self, n, m, mlp: nn.Module):
        """
        Linear policy class with a linear gain.
        Notation:
        - theta: parameters of the policy in R^d.
        - x: state in R^n
        - w: noise in R^m

        For linear policies, d = n * m (# of entries in gain matrix.)
        """
        super().__init__()
        # Compute dimension of the policy.
        self.n = n
        self.m = m
        self.mlp = mlp
        self.d = len(mlp.parameters())

    def evaluate_policy(self, x, theta):
        assert(len(theta) == self.d)
        return self.mlp(x)

    def evaluate_policy_batch(self, x_batch, theta):
        """
        input:
        - theta: shape (d)
        - x_batch: shape (B, n)
        output:
        - y : shape (B, m)
        """
        assert(x_batch.shape[1] == self.n)
        assert(len(theta) == self.d)
        return self.mlp(x_batch)

    def policy_jacobian(self, x, theta):
        eval = lambda theta_ad: self.evaluate_policy(x, theta_ad)
        return torch.autograd.functional.jacobian(eval, theta)

    def policy_jacobian_batch(self, x_batch, theta):
        eval = lambda theta_ad: self.evaluate_policy_batch(x_batch, theta_ad)
        return torch.autograd.functional.jacobian(eval, theta)

def test_nn_policy():
    policy_net = PolicyNLP(5,3)

    print(policy_net.state_dict())
    policy = NNPolicy(5,3, policy_net)

    print(policy.evaluate_policy(torch.rand(5), torch.rand(15)).shape)
    print(policy.evaluate_policy_batch(torch.rand(10, 5), torch.rand(15)).shape)

    print(policy.policy_jacobian(torch.ones(5), torch.ones(15)))
    print(policy.policy_jacobian_batch(torch.ones(10, 5), torch.ones(15)).shape)