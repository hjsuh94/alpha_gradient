import numpy as np

def compute_mean(x):
    """
    Given a batch of vector quantities, determine the mean.
    args: x of shape (B,n), batch of samples. 
    """
    B = x.shape[0]
    return np.sum(x, axis=0) / B

def compute_covariance_norm(x, p=2):
    """
    Given a batch of vector quantities, compute the covariance
    and computes a norm on the covariance matrix.
    args: x of shape (B,n), batch of samples.
    """
    B = x.shape[0]
    mu = compute_mean(x)
    deviations = np.subtract(x, mu) # B x n
    covariance = np.zeros((x.shape[1], x.shape[1]))
    for i in range(B):
        covariance += np.outer(deviations[i], deviations[i])
    covariance /= B
    return np.sqrt(np.linalg.norm(covariance, p))# / x.shape[1])

def compute_variance_norm(x, p=2):
    """
    Given a batch of vector quantities, determine the variance.
    Unlike the covariance norms, this computes the inner product
    and takes a vector norm. (i.e. only computes vector norm on 
    the diagonal elemetns of the covariance matrix).
    args: x of shape (B,n), batch of samples.
    """
    B = x.shape[0]
    mu = compute_mean(x)
    deviations = np.subtract(x, mu) # B x n
    covariance = np.zeros((x.shape[1], x.shape[1]))
    for i in range(B):
        covariance += np.dot(deviations[i], deviations[i])
    covariance /= B
    return np.sqrt(np.linalg.norm(covariance, p))# / x.shape[1])
