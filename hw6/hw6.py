import numpy as np
from scipy.stats import kstest, expon
from utils import seed_everything

seed_everything()

# Define the data
data = [122, 133, 106, 128, 135, 126]

# Calculate the mean of the data
mean_data = np.mean(data)

# Define the exponential distribution with the mean of the data as the loc parameter
exp_dist = expon(loc=mean_data)

# Perform the KS test
kstest_result = kstest(data, exp_dist.cdf)

# Print the results
print(f"KS test statistic: {kstest_result.statistic:.4f}")
print(f"p-value: {kstest_result.pvalue:.4f}")

