import numpy as np
import matplotlib.pyplot as plt
import torch

from alpha_gradient.objective_function import ObjectiveFunction
from alpha_gradient.dynamical_system import DynamicalSystem


class RobotMap:
    def __init__(self, centers, radius):
        """
        centers: N x 2 centers of obstacles.
        radius: N radius specifications.
        """
        self.centers = centers
        self.radius = radius

        assert(self.centers.shape[0] == len(radius))
        self.N = len(radius)

    def in_collision(self, x):
        """
        Takes in a states x of shape n.
        Returns torch.bool if x is in collision.
        """

        normed_distance = torch.norm(x - self.centers, dim=1)
        collision = torch.any(normed_distance < self.radius)
        return collision

    def in_collision_batch(self, x_batch):
        """
        Takes in a batch of states shape B x n
        Returns torch.bool that is B dimensional.
        """
        # Needs to be B x N array.
        normed_distance = torch.norm(x_batch.unsqueeze(1) - self.centers, dim=2)
        collision = torch.any(normed_distance < self.radius, dim=1)
        return collision

    def plot_map(self, ax):
        for i in range(self.N):
            circle = plt.Circle((self.centers[i,0], self.centers[i,1]),
                self.radius[i], color='b', alpha=0.1)
            ax.add_patch(circle)


def test_robot_map():
    centers = torch.tensor([
        [-1,0,1], 
        [0,np.sqrt(3),0]
    ]).transpose(0,1)
    radius = torch.tensor([0.3, 0.3, 0.3])

    robot_map = RobotMap(centers, radius)

    print(robot_map.in_collision(torch.tensor([-1.0, 0.0])))
    print(robot_map.in_collision(torch.tensor([1.0, 0.2])))
    print(robot_map.in_collision(torch.tensor([0.0, 1.9])))
    print(robot_map.in_collision(torch.tensor([0.0, 1.0])))
    print(robot_map.in_collision(torch.tensor([0.0, 0.0])))

    x_batch = torch.tensor([
        [1.0, 0.2], [-1.0, 0.0], [0.0, 1.9], [0.0, 1.0], [0.0, 0.0]])
    print(robot_map.in_collision_batch(x_batch))
    print(x_batch[robot_map.in_collision_batch(x_batch)])


    plt.figure()
    robot_map.plot_map(plt.gca())
    plt.axis('equal')
    plt.show()
