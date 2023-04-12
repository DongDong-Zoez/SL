import numpy as np
import random 

def seed_everything():
    np.random.seed(311657007)
    random.seed(311657007)

def box_muller_transform(u1, u2):
    """
    Generate n samples from a standard normal distribution
    using the Box-Muller transform.

    main idea:
        1. use the relationship inherit from box-muller transformation

    pseudo code:
        1. Random Sample U1, U2 from Uinf(0,1)
        2. calculate the formula (box-muller)
        3. Repeat above step N times

    annotations:
    """
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return z1, z2

def generate_multinormal_uniform(mean, cov, size=1):
    """
    Generate multivariate normal random variables using uniform random variables.

    Parameters:
        mean (array): Mean vector.
        cov (array): Covariance matrix.
        size (int): Number of samples to generate.

    Returns:
        array: Generated samples.
    """
    n = len(mean)
    assert n == len(cov)
    # Get Cholesky decomposition of covariance matrix
    L = np.linalg.cholesky(cov)

    # Generate uniform random variables
    u = np.random.uniform(size=(2, size * n))

    # Generate standard normal random variables using Box-Muller transform
    z = np.array([np.apply_along_axis(box_muller_transform, 0, u[0], u[1])]).reshape(n, -1)

    # Multiply Cholesky decomposition by standard normal random variables
    x = np.dot(L, z)

    # Add mean vector to result
    x = x + mean[:, None]

    return x.T