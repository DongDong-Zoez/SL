import numpy as np
import matplotlib.pyplot as plt

# Define parameters
lam1 = 1
lam2 = 2
lam3 = 3

# Generate 104 data points
n = 10000
X = np.random.exponential(scale=1/lam1, size=n)
Y = np.zeros(n)

for i in range(n):
    temp = np.array([X[i], X[(i+1)%n]/lam2, X[(i+2)%n]/lam3])
    Y[i] = np.min(temp)

# Plot rank plot
plt.plot(np.argsort(X)+1, np.argsort(Y)+1, 'o')
plt.plot([1, n], [1, n], 'k--')
plt.xlabel('Rank of X')
plt.ylabel('Rank of Y')
plt.title('Rank plot of Marshall-Olkin vector')
plt.savefig("assets/test.jpg")
