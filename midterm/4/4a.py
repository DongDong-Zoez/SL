import numpy as np
from scipy.stats import binom

# given data
data = np.array([6, 7, 3, 4, 7, 3, 7, 2, 6, 3, 7, 8, 2, 1, 3, 5, 8, 7])
n = len(data)

# define function to compute KS statistic
def ks_statistic(data, p):
    x = np.sort(data)
    n = len(x)
    F = binom.cdf(x, n=8, p=p)
    D = np.max(np.abs(F - np.arange(1, n+1)/n))
    return D

# compute KS statistic for a range of values of p
p_vals = np.mean(data) / 8
D_vals = ks_statistic(data, p_vals)

# print the maximum KS statistic and corresponding p value
print("KS statistic:", D_vals)
print("p with KS statistic:", p_vals)

# find the maximum KS statistic and corresponding p value
p_vals = np.linspace(0, 1, 101)
D_vals = [ks_statistic(data, p) for p in p_vals]
min_D = np.min(D_vals)
idx_min_D = np.argmin(D_vals)
p_min_D = p_vals[idx_min_D]

# print the maximum KS statistic and corresponding p value
print("Minimum KS statistic:", min_D)
print("p with minimum KS statistic:", p_min_D)