import numpy as np
import matplotlib.pyplot as plt

from utils import seed_everything
seed_everything()

# Define the dataset
data = np.array([0.0839, 0.0205, 0.3045, 0.7816, 0.0003,
                 0.0095, 0.4612, 0.9996, 0.9786, 0.7580,
                 0.0002, 0.7310, 0.0777, 0.4483, 0.4449,
                 0.7943, 0.1447, 0.0431, 0.8621, 0.3273])

# Define the sample size
n = len(data)

# Calculate the sample median
sample_median = np.median(data)

# Create n jackknife samples
jackknife_medians = np.zeros(n)
for i in range(n):
    jackknife_sample = np.delete(data, i)
    jackknife_medians[i] = np.median(jackknife_sample)

# Calculate the bias of the sample median
bias_median = (n - 1) * (np.mean(jackknife_medians) - sample_median)

# Calculate the variance of the sample median
var_median = (n - 1) / n * np.sum((jackknife_medians - np.mean(jackknife_medians)) ** 2)

# Calculate the standard error of the sample median
se_median = np.sqrt(var_median)

# Calculate the z-score for the desired confidence level (e.g., 95%)
alpha = 0.05
z = 1.96

# Calculate the confidence interval
lower = sample_median - z * se_median
upper = sample_median + z * se_median

# Plot the histogram of the jackknife sample medians
plt.hist(jackknife_medians, bins=10, edgecolor='black')
plt.axvline(sample_median, color='red', label='Sample median')
plt.axvline(np.mean(jackknife_medians), color='blue', label='Jackknife mean')
plt.xlabel('Median')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("assets/3a.jpg")

print('Sample median: {:.4f}'.format(sample_median))
print('Bias of the sample median: {:.4f}'.format(bias_median))
print('Variance of the sample median: {:.4f}'.format(var_median))
print('Standard error of the sample median: {:.4f}'.format(se_median))
print('95% Confidence interval: [{:.4f}, {:.4f}]'.format(lower, upper))
