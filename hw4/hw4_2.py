import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../")

from hw4_1 import generate_multinormal_uniform
import utils

utils.seed_everything()

mean = np.zeros(3)
cov = np.ones(9) * .5
cov[::4] = 1
cov = cov.reshape(3, -1)
samples = generate_multinormal_uniform(mean, cov, size=5000)

norm = stats.norm()
x_unif = norm.cdf(samples)

class Exp:

    def __init__(self, lam):
        self.lam = lam

    def pdf(self, x):
        return self.lam * np.exp(-self.lam * x)
    
    def ppf(self, x):
        return - np.log(1 - x)

exp = Exp(1)
x1_trans = exp.ppf(x_unif[:, 0])
x2_trans = exp.ppf(x_unif[:, 1])
x3_trans = exp.ppf(x_unif[:, 2])

print("Before Sample mean:", np.mean(samples, axis=0))
print("Before Sample covariance:\n", np.cov(samples, rowvar=False))

samples[:,0] = x1_trans
samples[:,1] = x2_trans
samples[:,2] = x3_trans

print("After Sample mean:", np.mean(samples, axis=0))
print("After Sample covariance:\n", np.cov(samples, rowvar=False))
print("Y1 mean:", np.mean(x1_trans))
print("Y1 covariance:\n", np.cov(x1_trans, rowvar=False))
print("Y2 mean:", np.mean(x2_trans))
print("Y2 covariance:\n", np.cov(x2_trans, rowvar=False))

h = sns.jointplot(x=x1_trans, y=x2_trans, kind='kde')
h.set_axis_labels('Y1', 'Y2', fontsize=16)
plt.savefig("assets/multi-expon-x.png") # save fig file

Y1 = np.argsort(x1_trans)
Y2 = np.argsort(x2_trans)

h = sns.jointplot(x=Y1, y=Y2, kind='hex')
h.set_axis_labels('Y1', 'Y2', fontsize=16)
plt.savefig("assets/rank-plot.png") # save fig file