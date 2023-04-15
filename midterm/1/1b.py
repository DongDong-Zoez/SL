from utils import dirichlet, seed_everything
import matplotlib.pyplot as plt
from scipy.special import gamma

import numpy as np

seed_everything()

alpha = np.array([2,3,4])  

def compute_M(grid):
    res = np.apply_along_axis(lambda x: dirichlet(x), 1, grid)
    return max(res)

grid = np.random.uniform(0,1,3*10000).reshape(10000,3)
grid = np.apply_along_axis(lambda x: x / x.sum(), 1, grid)

def accept_rejection_algorithm(pdf, M, dims=3):
    while True:
        x = np.random.uniform(0, 1, dims)
        x = x / x.sum()
        y = np.random.uniform(0, 1)
        if y*M <= pdf(x):
            return x
      
M = compute_M(grid)
dirichlet_sample = lambda: accept_rejection_algorithm(dirichlet, M)
smpls = np.array([dirichlet_sample() for _ in range(10000)])
print("Samples Mean:", np.mean(smpls, axis=0))
print("Samples Covariance Matrix:", np.cov(smpls.T))

for i in range(smpls.shape[1]):
    plt.figure()
    plt.hist(smpls[:,i], bins=50, edgecolor='black')
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.savefig(f"assets/1b_x{i}.jpg")

print(np.mean(smpls, axis=0))
