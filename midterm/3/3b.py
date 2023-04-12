import numpy as np
import matplotlib.pyplot as plt

from utils import seed_everything
seed_everything()

# Define the dataset
data = np.array([0.0839, 0.0205, 0.3045, 0.7816, 0.0003,
                 0.0095, 0.4612, 0.9996, 0.9786, 0.7580,
                 0.0002, 0.7310, 0.0777, 0.4483, 0.4449,
                 0.7943, 0.1447, 0.0431, 0.8621, 0.3273])

# Define the number of bootstrap samples to create
B = 10**4

# Define the sample size
n = len(data)

# Calculate the sample median
sample_median = np.median(data)

# Create B bootstrap samples
bootstrap_medians = np.zeros(B)
for i in range(B):
    bootstrap_sample = np.random.choice(data, size=n, replace=True)
    bootstrap_medians[i] = np.median(bootstrap_sample)

# Calculate the bias of the sample median
bias_median = np.mean(bootstrap_medians) - sample_median

# Calculate the variance of the sample median
var_median = np.mean((bootstrap_medians - np.mean(bootstrap_medians)) ** 2)

# Calculate the standard error of the sample median
se_median = np.sqrt(var_median)

# Calculate the z-score for the desired confidence level (e.g., 95%)
alpha = 0.05
z = 1.96

# Calculate the confidence interval
lower = sample_median - z * se_median
upper = sample_median + z * se_median

# Plot the histogram of the bootstrap sample medians
plt.hist(bootstrap_medians, bins=10, edgecolor='black')
plt.axvline(sample_median, color='red', label='Sample median')
plt.axvline(np.mean(bootstrap_medians), color='blue', label='Bootstrap mean')
plt.xlabel('Median')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("assets/3b.jpg")

print('Sample median: {:.4f}'.format(sample_median))
print('Bias of the sample median: {:.4f}'.format(bias_median))
print('Variance of the sample median: {:.4f}'.format(var_median))
print('Standard error of the sample median: {:.4f}'.format(se_median))
print('95% Confidence interval: [{:.4f}, {:.4f}]'.format(lower, upper))
