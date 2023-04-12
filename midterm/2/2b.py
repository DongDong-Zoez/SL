import numpy as np
import matplotlib.pyplot as plt

from utils import seed_everything
seed_everything()

# Set parameters
n = 10000
lambdas = [1, 2, 3]

def rCopula(n, lambda_):
    samples = []
    for _ in range(n):
        t1 = np.random.exponential(1/lambda_[0])   # T_1 ~ Exp(lambda_1)
        t2 = np.random.exponential(1/lambda_[1])   # T_2 ~ Exp(lambda_2)
        t3 = np.random.exponential(1/lambda_[2])   # T_3 ~ Exp(lambda_3)
        x = min(t1, t3)   # P = min(T_1, T_3)
        y = min(t2, t3)   # Q = min(T_2, T_3)
        samples.extend([x, y])
    samples = np.array(samples).reshape(n, 2)
    return samples


samples = rCopula(n, lambdas)

X, Y = samples[...,0], samples[...,1]

print("Samples Mean:", np.mean(samples, axis=0))
print("Samples Covariance Matrix:", np.cov(samples.T))

# Plot rank plot
fig, ax = plt.subplots()
ax.plot(np.argsort(X).argsort(), np.argsort(Y).argsort(), "o")
ax.set_xlabel('Rank of X')
ax.set_ylabel('Rank of Y')
ax.set_title('Rank Plot')
plt.savefig("assets/2b.jpg")
