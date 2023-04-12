import numpy as np
from scipy.stats import beta

from utils import seed_everything
seed_everything()

# Input data
data = np.array([[0.0839, 0.0205, 0.3045, 0.7816, 0.0003],
                 [0.0095, 0.4612, 0.9996, 0.9786, 0.7580],
                 [0.0002, 0.7310, 0.0777, 0.4483, 0.4449],
                 [0.7943, 0.1447, 0.0431, 0.8621, 0.3273]])

# Estimate the parameters of the beta distribution
a, b, _, _ = beta.fit(data.flatten())

# Generate 10^4 random samples from the estimated beta distribution
n_samples = 10**4
bootstrap_samples = beta.rvs(a, b, size=(n_samples, data.shape[0], data.shape[1]))

# Calculate the sample medians for each bootstrap sample
sample_medians = np.median(bootstrap_samples, axis=(1, 2))

# Calculate the mean and standard deviation of the sample medians
mean_sample_medians = np.mean(sample_medians)
std_sample_medians = np.std(sample_medians, ddof=1)
# std_sample_medians = np.mean((sample_medians - np.mean(sample_medians))**2) ** 0.5


# Construct the 95% confidence interval
lower_bound = mean_sample_medians - 1.96 * std_sample_medians
upper_bound = mean_sample_medians + 1.96 * std_sample_medians
print(f"95% Confidence Interval: ({lower_bound}, {upper_bound})")
import matplotlib.pyplot as plt

plt.hist(sample_medians, bins=30)
plt.xlabel('Sample Medians')
plt.ylabel('Frequency')
plt.title('Histogram of Bootstrap Sample Medians')
plt.savefig("assets/3c.jpg")
