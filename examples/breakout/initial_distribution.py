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
    ball_x0 = 1.0 * torch.ones(sample_size) + torch.normal(
        0, 1.0, (sample_size,1)).squeeze(1)
    ball_y0 = 2.0 * torch.ones(sample_size) + torch.normal(
        0, 0.2, (sample_size,1)).squeeze(1)
    ball_vx0 = -0.2 * ball_x0 + torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)
    ball_vy0 = -0.2 * ball_y0 + torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)

    pad_x0 = torch.normal(0.0, 0.5, (sample_size,1)).squeeze(1)
    pad_y0 = torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)
    pad_theta0 = torch.normal(0.0, 0.01, (sample_size,1)).squeeze(1)

    return torch.vstack(
        (ball_x0, ball_y0, ball_vx0, ball_vy0, pad_x0, pad_y0, pad_theta0)
        ).transpose(0,1)
