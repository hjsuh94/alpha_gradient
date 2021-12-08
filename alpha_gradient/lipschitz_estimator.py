import numpy as np
from scipy.optimize import fsolve


"""1. Functions for Computing sample Lipschitz constants."""

def compute_pairwise_lipschitz_matrix(x, y, p=2):
    """
    Given a (B x n) array of x and (B x 1) array of y, compute the 
    pairwise Lipschitz computation |y_i - y_j| / |x_i - x_j|_p 
    and return it as a symmetric matrix.
    """
    B = x.shape[0]
    # Do not ask me how I got this....
    pairwise_x_norm = np.linalg.norm(
        np.subtract.outer(x,x).diagonal(axis1=1,axis2=3), ord=p, axis=2)

    # If someone passes a vector isntead of dim 2m atrix.
    if len(y.shape) == 1:
        y = y[:,None]

    # This norm just takes the absolute value.
    pairwise_y_norm = np.linalg.norm(
        np.subtract.outer(y,y).diagonal(axis1=1,axis2=3), ord=1, axis=2)
    # The eye in the denominator is there to prevent nan values along the 
    # diagonal. The diagonal of pairwise_L will evaluate to zero since
    # the diagonal of pairwise_y_norm and pairwise_x_norm are both zero.
    pairwise_L = pairwise_y_norm / (pairwise_x_norm + np.eye(B))
    return pairwise_L

# A convenience function just to turn the pairwise L matrix as vector.
def compute_pairwise_lipschitz_vector(x, y, p=2):
    """
    Given a (B x n) array of x and (B x 1) array of y, compute the 
    pairwise Lipschitz computation |y_i - y_j| / |x_i - x_j|_p 
    and return it as a symmetric matrix.
    """
    pairwise_L_mat = compute_pairwise_lipschitz_matrix(x, y, p)
    indices = np.triu_indices_from(pairwise_L_mat)
    pairwise_vec = np.asarray(pairwise_L_mat[indices])
    return pairwise_vec

def compute_pairwise_lipschitz_tensor(x, y, p=2):
    """
    Given a (T x B x n) array of x and (T x B x 1) array of y, compute the 
    pairwise Lipschitz computation |y_i - y_j| / |x_i - x_j|_p 
    and return it as a symmetric matrix.
    """
    T = x.shape[0]
    B = x.shape[1]
    n = x.shape[2]    
    x_flatten = np.reshape(x, (T*B, n))
    y_flatten = np.reshape(y, (T*B, 1))
    L_flatten = compute_pairwise_lipschitz_vector(x_flatten, y_flatten, p)
    max_ind = int(np.divide(len(L_flatten), T))
    return np.reshape(L_flatten[:T*max_ind], (T, int(len(L_flatten) / T), 1))

"""2. Functions for Frechet distribution fits."""

def frechet_pdf(alpha, sigma, x):
    """
    Express pdf of two-parameter family Frechet distribution given list of x.
    """
    a = alpha / sigma
    b = np.power(x / sigma, -1 - alpha)
    c = np.exp(-np.power((x/sigma), -alpha))
    return a * b * c

def frechet_cdf(alpha, sigma, x):
    """
    Express cdf of two-parameter family Frechet distribution given list of x.
    Gives: What is the probability that the estimate is less than x?
    """    
    return np.exp(-np.power(x/sigma, -alpha))

"""3. Functions for MLE using Frechet distributions."""
"""
The methods here are derived from [1] "Maximum likelihood estimation for the
Frechet distribution based on block maxima extracted from a time series"
by Bucher and Segers.
"""

def psi(alpha, x):
    """
    Given scalar alpha and a (B) array of x, compute the psi function.
    """
    k = len(x)
    x[x == 0] = 1e-1    
    a = 1 / alpha 
    b = (1/k) * np.sum(np.power(x, -alpha) * np.log(x))
    c = (1/k) * np.sum(np.power(x, -alpha))
    d = (1/k) * np.sum(np.log(x))
    return a + b / c - d

def sigma(alpha, x):
    """
    Given scalar alpha and a (B) array of x, compute the sigma function.
    """
    k = len(x)
    return np.power((1/k) * np.sum(np.power(x, -alpha)), -1/alpha)    

def frechet_mle(data_lst):
    """
    Solve the MLE problem given data to fit a Frechet distribution, given a
    length B array of scalars. 
    """
    # From [1], we know that psi is a monotonic function with a unique
    # root. Computation of fsolve usually takes takes around 0.5ms.
    if (np.max(data_lst) == np.min(data_lst)):
        alpha_star = None
        sigma_star = None
    else:
        alpha_star = fsolve(lambda alpha: psi(alpha, data_lst), 0.1)
        sigma_star = sigma(alpha_star, data_lst)
    return (alpha_star, sigma_star)

"""4. Wrapper functions for convenience."""
def estimate_frechet_parameters(X, y):
    """
    X should be a (T x B x n) array of samples. y is (T x B x 1). 
    T will be used to construct samples to fit Frechet distribution from.
    B will be used to compute subsamples to take Lipschitz constant from.
    n is the dimension of the underlying vector space.

    Note that to not waste computation, we compute a very dense Lipschitz
    batch computation on (T x B x n), so B can be typically set to a low 
    value.
    """
    trial_mat = compute_pairwise_lipschitz_tensor(X, y)
    trial_lst = np.max(trial_mat, axis=1)
    alpha_star, sigma_star = frechet_mle(trial_lst)
    return alpha_star, sigma_star

def estimate_lipschitz_probability(X, y, Lbar):
    """
    X should be a (T x B x n) array of samples. y is (T x B x 1). 
    T will be used to construct samples to fit Frechet distribution from.
    B will be used to compute subsamples to take Lipschitz constant from.
    n is the dimension of the underlying vector space.

    Note that to not waste computation, we compute a very dense Lipschitz
    batch computation on (T x B x n), so B can be typically set to a low 
    value.
    """
    alpha_star, sigma_star = estimate_frechet_parameters(X, y)
    if (alpha_star == None):
        return 1.0
    else: 
        return frechet_cdf(alpha_star, sigma_star, Lbar)
