import torch

def softplus_batch_positive(x_batch, stiffness, kappa):
    return (stiffness / kappa) * (
        torch.log(1 + torch.exp(
            -kappa * x_batch)) + kappa * x_batch)

def softplus_batch_negative(x_batch, stiffness, kappa):
    return (stiffness / kappa) * (
        torch.log(1 + torch.exp(
            kappa * x_batch)))

def softplus(x, stiffness, kappa):
    if x >= 0:
        f = softplus_batch_positive(x, stiffness, kappa)
    else:
        f = softplus_batch_negative(x, stiffness, kappa)
    return f 

def softplus_batch(x_batch, stiffness, kappa):
    batch_pos_ind = x_batch >= 0
    batch_neg_ind = x_batch < 0
    f_batch = torch.zeros(x_batch.shape[0])
    f_batch[batch_pos_ind] = softplus_batch_positive(
        x_batch[batch_pos_ind], stiffness, kappa)
    f_batch[batch_neg_ind] = softplus_batch_negative(
        x_batch[batch_neg_ind], stiffness, kappa)
    return f_batch
