# 統計學習第六次作業

在這次的作業中，要求逼近檢定一個資料是否來自指數分配的 P 值，我們採用 KS 檢定來判別給定的資料是否屬於指數分配

Follow up the instruction, you may see the result

```
KS test statistic: 0.4502
p-value: 0.1251
```

## Scipy ketest

```python
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
```

In this report, we have the following steps

1. Define our data ```[122, 133, 106, 128, 135, 126]```
2. For examinating if the data are from the exp dist., we use the mean of the data as the location parameter of the theoricatal distribution parameter 
3. perform ```kstest```
4. Output our result in the terminal