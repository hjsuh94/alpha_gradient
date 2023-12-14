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

def relu(x, stiffness):
    if x >= 0:
        f = stiffness * x
    else:
        f = 0.0
    return f

def softplus_batch(x_batch, stiffness, kappa):
    """
    batch_pos_ind = x_batch >= 0
    batch_neg_ind = x_batch <= 0
    f_batch = torch.zeros(x_batch.shape[0])
    f_batch[batch_pos_ind] = softplus_batch_positive(
        x_batch[batch_pos_ind], stiffness, kappa)
    f_batch[batch_neg_ind] = softplus_batch_negative(
        x_batch[batch_neg_ind], stiffness, kappa)
    """
    
    softplus = torch.nn.Softplus(beta=kappa, threshold=20)
    f_batch = stiffness * softplus(x_batch)
    return f_batch

def relu_batch(x_batch, stiffness):
    batch_pos_ind = x_batch >= 0
    batch_neg_ind = x_batch < 0
    f_batch = torch.zeros(x_batch.shape[0])
    f_batch[batch_pos_ind] = stiffness * x_batch[batch_pos_ind]
    f_batch[batch_neg_ind] = 0.0
    return f_batch
