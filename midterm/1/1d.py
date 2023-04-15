import numpy as np
import matplotlib.pyplot as plt

alpha = np.array([2,3,4])

def gamma_sample(shape, scale=1, size=1, max_iter=10000):
    """
    Generate random samples from the gamma distribution with shape parameter `shape`
    and scale parameter `scale` using the acceptance-rejection method.

    Parameters
    ----------
    shape : float
        Shape parameter of the gamma distribution.
    scale : float, optional
        Scale parameter of the gamma distribution.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k),
        then m * n * k samples are drawn.
    max_iter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    ndarray
        Random samples from the gamma distribution with shape (size,).
    """
    x = np.zeros(size)
    for i in range(size):
        # Rejection sampling loop
        for j in range(max_iter):
            # Generate candidate sample from uniform distribution
            y = np.random.uniform()
            # Calculate the candidate sample
            candidate = -np.log(y)
            if np.random.uniform() <= candidate ** (shape - 1) * np.exp(-candidate / scale) / (np.math.gamma(shape) * scale ** shape):
                break
        x[i] = candidate
    return x

def dirichlet_sample(alpha, size=10000):
    """
    Generate random samples from the Dirichlet distribution with parameter vector alpha.

    Parameters
    ----------
    alpha : array-like
        Parameter vector of length K.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k),
        then m * n * k samples are drawn.

    Returns
    -------
    ndarray
        Random samples from the Dirichlet distribution with shape (size, K).
    """
    alpha = np.array(alpha)
    if alpha.ndim == 0:
        alpha = alpha[np.newaxis]
    x = np.zeros((size, len(alpha)))
    for i in range(size):
        # Generate gamma samples using acceptance-rejection method
        gammas = np.array([gamma_sample(a) for a in alpha])
        x[i] = gammas.squeeze(-1) / np.sum(gammas)
    return x


smpls = np.array(dirichlet_sample(alpha))
print("Samples Mean:", np.mean(smpls, axis=0))
print("Samples Covariance Matrix:", np.cov(smpls.T))

for i in range(smpls.shape[1]):
    plt.figure()
    plt.hist(smpls[:,i], bins=50, edgecolor='black')
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.savefig(f"assets/1d_x{i}.jpg")