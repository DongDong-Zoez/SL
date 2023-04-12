import numpy as np
from scipy.special import gamma
import random


def seed_everything():
    np.random.seed(311657007)
    random.seed(311657007)


def trimmed_mean(x, trim_pct=0.2):
    n_trim = int(np.ceil(len(x) * trim_pct))
    return np.mean(x[n_trim:-n_trim])


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
    z = np.array([np.apply_along_axis(
        box_muller_transform, 0, u[0], u[1])]).reshape(n, -1)

    # Multiply Cholesky decomposition by standard normal random variables
    x = np.dot(L, z)

    # Add mean vector to result
    x = x + mean[:, None]

    return x.T


def accept_rejection_algorithm_hook(pdf, *args, **kwargs):
    print(args, kwargs)
    try:
        dims = kwargs["dims"]
    except:
        dims = 1
    print(dims)
    def ar_algorithm(M):
        while True:
            x = np.random.uniform(0, 1, dims)
            y = np.random.uniform(0, 1, dims)
            if y*M <= pdf(x):
                return x
    def wrapper(*args, **kwargs):
        try: 
            M = kwargs["M"]
            num_smpls = kwargs["num_smpls"]
        except:
            raise KeyError("Miss keyword arguments, have you set M or num_smpls??")
        smpls = list(map(ar_algorithm, [M for _ in range(num_smpls)]))
        return smpls
    return wrapper

def dirichlet(x, alpha=np.array([2,3,4])):
    alpha_sum = sum(alpha)
    alphas = gamma(alpha)
    alphas = np.prod(alphas)
    return np.prod(x ** (alpha - 1)) * gamma(alpha_sum) / alphas

# @accept_rejection_algorithm_hook
# def pdf(x, *args, **kwargs):
#     return x + 0.5
# samples = pdf(M=5, num_smpls=10000)
# plt.hist(samples, density=True)
# plt.savefig("result.jpg")