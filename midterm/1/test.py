import math
import random
import numpy as np

from utils import dirichlet as dirichlet_pdf

# Define the Dirichlet distribution parameters
alpha = np.array([2, 3, 4])

# Define the proposal distribution
g = lambda n: [[random.uniform(0, 1) for i in range(len(alpha))] for j in range(n)]

def acceptance_prob(x):
    """
    Calculate the acceptance probability for a proposal x from the proposal distribution.
    """
    return dirichlet_pdf(x, alpha) / dirichlet_pdf(x, [1]*len(alpha))

def sample_dirichlet(alpha, n):
    """
    Generate n samples from the Dirichlet distribution with parameter vector alpha using the acceptance-rejection method.
    """
    # Initialize the output array
    samples = [[0 for i in range(len(alpha))] for j in range(n)]

    # Generate samples
    i = 0
    while i < n:
        # Generate a proposal sample
        x = g(1)[0]

        # Accept or reject the proposal
        u = random.uniform(0, 1)
        if u <= acceptance_prob(x):
            samples[i] = x
            i += 1

    return samples


sample_dirichlet(alpha, 100)
