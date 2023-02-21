import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)

def g1(x):
    return 1

def g2(x):
    return 2 * x

def G1inv(x):
    return x

def G2inv(x):
    return x ** 0.5

def pdf(x):
    p1 = 0.5
    p2 = 0.5
    return p1 * g1(x) + p2 * g2(x)

def composition_method(x,p):
    r = np.random.uniform(0,1)
    if x < p:
        return G1inv(r)
    else:
        return G2inv(r)

def main():
    """
    PDF: .5 + x

    main idea:
        1.  g1 = .5 * c1, g2 = x * c2
        2. make g1, g2 pdf --> (c1, c2) = (.5, .5)
        3. calculate inverse cdf of g1, g2. x, x**0.5, respectively
        4. calculate PDF = .5 * g1 + .5 * g2 --> (p1, p2) = (.5, .5)

    pseudo code:
        1. Random Sample U1, U2 from Uinf(0,1)
        2. If U1 > p1, then drawSample(G1inv(U2)), else drawSample(G2inv(U2))   
        3. Repeat above step N times

    annotations:
        1. darwSample: a function to obtain sample
        2. G1inv: the inverse CDF of g1
        3. G2inv: the inverse CDF of g2 
    """
    np.random.seed(42)
    p = 0.5
    rand = np.random.uniform(0,1,10000)
    x = np.linspace(0,1,100)
    y = pdf(x)

    samples = [composition_method(r, p) for r in rand]

    plt.figure()
    plt.title("Composition Method")

    plt.plot(x, y, label="f(x)")
    plt.hist(samples, bins=50,label='sampling', density=True)
    plt.legend()
    plt.savefig("assets/1_c.jpg")

if __name__ == "__main__":
    main()
