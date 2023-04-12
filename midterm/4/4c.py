import numpy as np
from scipy.stats import binom

# Generate bootstrap samples
n_samples = int(1e4)
original_data = np.array([6, 7, 3, 4, 7, 3, 7, 2, 6, 3, 7, 8, 2, 1, 3, 5, 8, 7])
p_vals = np.mean(original_data) / 8
boot_samples = np.random.choice(original_data, size=(len(original_data), n_samples), replace=True)

boot_samples = np.random.binomial(8,p_vals,18*n_samples).reshape(10000,18)

def ks_statistic(data, p=p_vals):
    x = np.sort(data)
    n = len(x)
    F = binom.cdf(x, n=8, p=p)
    D = np.max(np.abs(F - np.arange(1, n+1)/n))
    return D

# Compute Kolmogorov-Smirnov statistic for each bootstrap sample
boot_stats = np.apply_along_axis(ks_statistic, axis=1, arr=boot_samples)
# Compute p-value
observed_stat = ks_statistic(original_data)
p_value = np.mean(boot_stats >= observed_stat)

print("Observed KS statistic:", observed_stat)
print("Booststrap KS statistic:", np.mean(boot_stats))
print("Bootstrap p-value:", p_value)
