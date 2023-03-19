# 統計學習第三次作業

在這次的作品中，我們實現了 box-muller 的常態分配模擬，除此之外，也實作了用 copula 模擬邊際隨機變數

## Normal Distribution

Write a program to simulate a three-dimensional normal distribution with mean zero and

- Var(X_i) = 1
- Corr(X_i, X_j) = 0.5

Simulate 10^4 points. Compute the mean, variance and correlation of your sample.

The answer is to use cholesky decomposition

For generating standard normal distribution, we apply box-muller transformation

```python
def box_muller_transform(u1, u2):
    """
    Generate n samples from a standard normal distribution
    using the Box-Muller transform.

    main idea:
        1. use the relationship inherit from box-muller transformation

    pseudo code:
        1. Random Sample U1, U2 from Uinf(0,1)
        2. calculate the formula (box-muller)
        3. Repeat above step N times

    annotations:
    """
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return z1, z2
```

then we can write down the generative multinormal dist. function

```python
def generate_multinormal_uniform(mean, cov, size=1):
    """
    Generate multivariate normal random variables using uniform random variables.

    Parameters:
        mean (array): Mean vector.
        cov (array): Covariance matrix.
        size (int): Number of samples to generate.

    Returns:
        array: Generated samples.
    """
    n = len(mean)
    assert n == len(cov)
    # Get Cholesky decomposition of covariance matrix
    L = np.linalg.cholesky(cov)

    # Generate uniform random variables
    u = np.random.uniform(size=(2, size * n))

    # Generate standard normal random variables using Box-Muller transform
    z = np.array([np.apply_along_axis(box_muller_transform, 0, u[0], u[1])]).reshape(n, -1)

    # Multiply Cholesky decomposition by standard normal random variables
    x = np.dot(L, z)

    # Add mean vector to result
    x = x + mean[:, None]

    return x.T
```

## Copula

Now simulate a three dimensional (Y_1, Y_2, Y_3) so that Y_i is marginally exp(1), and have the copula as same as the X in part 1. Simulate 10^4 points. Plot the rank plot between Y_1 and Y_2

The general framework of the algorithm (Gaussian copula) generating multivariate random vectors X with covariance is that

1. Generating multinormal random vectors W by cholesky decomposition with mean 0 and covariance equal to the vectors X
2. plug-in random vectors W into its CDF to make it a uniform random vectors U (The joint distribution is the Gaussian copula)
3. plug-in U into the Invert CDF of X 

```python
# define the mean vector and covariance structure
mean = np.zeros(3)
cov = np.ones(9) * .5
cov[::4] = 1
cov = cov.reshape(3, -1)

# generating random vectors W
samples = generate_multinormal_uniform(mean, cov, size=5000)

# generating random vectors U
norm = stats.norm()
x_unif = norm.cdf(samples)

# Target distribution
class Exp:

    def __init__(self, lam):
        self.lam = lam

    def pdf(self, x):
        return self.lam * np.exp(-self.lam * x)
    
    def ppf(self, x):
        return - np.log(1 - x)

# Generating ranodm vectors X
exp = Exp(1)
x1_trans = exp.ppf(x_unif[:, 0])
x2_trans = exp.ppf(x_unif[:, 1])
x3_trans = exp.ppf(x_unif[:, 2])
```

To verify if copula is the same, we have to check the rank plot, i.e., 

```python
# Calculating the rank
Y1 = np.argsort(x1_trans)
Y2 = np.argsort(x2_trans)

# rank plot
h = sns.jointplot(x=Y1, y=Y2, kind='hex')
h.set_axis_labels('Y1', 'Y2', fontsize=16)
plt.savefig("assets/rank-plot.png")
```

## Usage

The project is under Ubuntu 20.04

```
source ./script.sh
```




