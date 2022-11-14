import numpy as np
import torch

# Set up sampling function for x0.
def sample_x0_batch_narrow(sample_size):
    ball_x0 = 1.0 * torch.ones(sample_size) + torch.normal(
        0, 0.2, (sample_size,1)).squeeze(1)
    ball_y0 = 2.0 * torch.ones(sample_size) + torch.normal(
        0, 0.1, (sample_size,1)).squeeze(1)
    ball_vx0 = -0.2 * ball_x0 + torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)
    ball_vy0 = -0.2 * ball_y0 + torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)

    pad_x0 = torch.normal(0.0, 0.1, (sample_size,1)).squeeze(1)
    pad_y0 = torch.normal(0.0, 0.1, (sample_size,1)).squeeze(1)
    pad_theta0 = torch.normal(0.0, 0.1, (sample_size,1)).squeeze(1)

    return torch.vstack(
        (ball_x0, ball_y0, ball_vx0, ball_vy0, pad_x0, pad_y0, pad_theta0)
        ).transpose(0,1)

# Set up sampling function for x0.
def sample_x0_batch(sample_size):
    #ball_x0 = torch.zero(sample_size) + torch.rand(
    #    0, 1.0, (sample_size,1)).squeeze(1)
    ball_x0 = -2.0 * torch.ones(sample_size) + 4.0 * torch.rand(sample_size)
    ball_y0 = 2.0 * torch.ones(sample_size) + torch.normal(
        0, 0.2, (sample_size,1)).squeeze(1)
    ball_vx0 = -0.2 * ball_x0 + torch.normal(0.0, 0.03, (sample_size,1)).squeeze(1)
    ball_vy0 = -0.2 * ball_y0 + torch.normal(0.0, 0.03, (sample_size,1)).squeeze(1)

    pad_x0 = torch.normal(0.0, 0.5, (sample_size,1)).squeeze(1)
    pad_y0 = torch.normal(0.0, 0.05, (sample_size,1)).squeeze(1)
    pad_theta0 = torch.normal(0.0, 0.05, (sample_size,1)).squeeze(1)

    return torch.vstack(
        (ball_x0, ball_y0, ball_vx0, ball_vy0, pad_x0, pad_y0, pad_theta0)
        ).transpose(0,1)

# Set up sampling function for x0.
def sample_x0_batch_vert(sample_size):
    #ball_x0 = torch.zero(sample_size) + torch.rand(
    #    0, 1.0, (sample_size,1)).squeeze(1)
    ball_x0 = -1.25 * torch.ones(sample_size) + 2.5 * torch.rand(sample_size)
    ball_y0 = 2.0 * torch.ones(sample_size) + torch.normal(
        0, 0.2, (sample_size,1)).squeeze(1)
    ball_vx0 = 0.0 + torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)
    ball_vy0 = -0.3 + torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)

    pad_x0 = torch.normal(0.0, 0.5, (sample_size,1)).squeeze(1)
    pad_y0 = torch.normal(0.0, 0.05, (sample_size,1)).squeeze(1)
    pad_theta0 = torch.normal(0.0, 0.05, (sample_size,1)).squeeze(1)

    return torch.vstack(
        (ball_x0, ball_y0, ball_vx0, ball_vy0, pad_x0, pad_y0, pad_theta0)
        ).transpose(0,1)        

def sample_x0_batch_single(sample_size):
    #ball_x0 = torch.zero(sample_size) + torch.rand(
    #    0, 1.0, (sample_size,1)).squeeze(1)
    ball_x0 = torch.zeros(sample_size)
    ball_y0 = 2.0 * torch.ones(sample_size)
    ball_vx0 = torch.zeros(sample_size)
    ball_vy0 = -0.2 * torch.ones(sample_size)

    pad_x0 = torch.normal(0.5, 0.0, (sample_size,1)).squeeze(1)
    pad_y0 = torch.normal(0.0, 0.00, (sample_size,1)).squeeze(1)
    pad_theta0 = torch.normal(0.0, 0.00, (sample_size,1)).squeeze(1)

    return torch.vstack(
        (ball_x0, ball_y0, ball_vx0, ball_vy0, pad_x0, pad_y0, pad_theta0)
        ).transpose(0,1)
