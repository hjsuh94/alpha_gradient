import numpy as np

def compute_mean(x):
    """
    Given a batch of vector quantities, determine the mean.
    args: x of shape (B,n), batch of samples. 
    """
    B = x.shape[0]
    return np.sum(x, axis=0) / B

def compute_variance(x, p=2):
    """
    Given a batch of vector quantities, determine the variance.
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

X = np.random.normal(0, 2.0, (1000, 5))
print(compute_variance(X))