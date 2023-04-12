from utils import dirichlet, seed_everything
import matplotlib.pyplot as plt
from scipy.special import gamma

import numpy as np

seed_everything()

alpha = np.array([2,3,4])  

def compute_M(alpha):
    M = gamma(sum(alpha)) / np.prod(gamma(alpha))
    return M

def accept_rejection_algorithm(pdf, M, dims=3):
    while True:
        x = np.random.uniform(0, 1, dims)
        x = x / x.sum()
        y = np.random.uniform(0, 1)
        if y*M <= pdf(x):
            return x
      
dirichlet_sample = lambda: accept_rejection_algorithm(dirichlet, compute_M(alpha))
smpls = np.array([dirichlet_sample() for _ in range(10000)])
print("Samples Mean:", np.mean(smpls, axis=0))
print("Samples Covariance Matrix:", np.cov(smpls.T))

for i in range(smpls.shape[1]):
    plt.figure()
    plt.hist(smpls[:,i], bins=10, edgecolor='black')
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.savefig(f"assets/1b_x{i}.jpg")

print(np.mean(smpls, axis=0))
