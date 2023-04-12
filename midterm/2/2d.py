import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from utils import seed_everything, generate_multinormal_uniform

seed_everything()

# Set parameters
n = 10000
lambdas = [1, 2, 3]

import numpy as np
from scipy.stats import norm

def rCopula(n, lambda_):
    samples = []
    for _ in range(n):
        t1 = np.random.exponential(1/lambda_[0])   # T_1 ~ Exp(lambda_1)
        t2 = np.random.exponential(1/lambda_[1])   # T_2 ~ Exp(lambda_2)
        t3 = np.random.exponential(1/lambda_[2])   # T_3 ~ Exp(lambda_3)
        x = min(t1, t3)   # P = min(T_1, T_3)
        y = min(t2, t3)   # Q = min(T_2, T_3)
        samples.extend([x, y])
    samples = np.array(samples).reshape(n, 2, order='F')
    return samples

def moCopula(size, mu_z, mu_w, sigma_z, sigma_w, lambda1, lambda2, lambda3):
    # Step 1
    samples = rCopula(size, [lambda1, lambda2, lambda3]) # Assuming rCopula is defined and returns X and Y
    X, Y = samples[...,0], samples[...,1]
    
    # Step 3
    phi = norm.ppf
    result = np.empty((size, 2))
    for i in range(size):
        x = np.exp(-(lambda1 + lambda3) * X[i])
        y = np.exp(-(lambda2 + lambda3) * Y[i])
        result[i] = [phi(x, loc=mu_z, scale=sigma_z), phi(y, loc=mu_w, scale=sigma_w)]

    return result



mean = np.array([0, -2])
sigma = np.array([1, 2])

samples = moCopula(n, *mean, *sigma, *lambdas)

Z = samples[...,0]
W = samples[...,1]

print("After Sample mean:", np.mean(samples, axis=0))
print("After Sample covariance:\n", np.cov(samples, rowvar=False))
print("Z mean:", np.mean(Z))
print("Z covariance:\n", np.cov(Z, rowvar=False))
print("W mean:", np.mean(W))
print("W covariance:\n", np.cov(W, rowvar=False))

sns.scatterplot(x=Z, y=W)
plt.savefig("assets/2d.jpg") # save fig file