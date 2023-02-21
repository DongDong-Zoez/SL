import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)


def pdf(x):
    return .5 + x

def f_inverse(x):
    return 0.5 * ((8 * x + 1) ** 0.5 - 1)

def main():
    """
    PDF: .5 + x

    main idea:
        1. calculate f inverse --> 0.5 * ((8 * x + 1) ** 0.5 - 1)

    pseudo code:
        1. Random Sample U from Uinf(0,1)
        2. drawSample(inverseF(U))
        3. Repeat above step N times

    annotations:
        1. darwSample: a function to obtain sample
        2. inverseF: a function calculate the inverse CDF of f
    """
    np.random.seed(42)
    x = np.linspace(0,1,100)
    y = pdf(x)
    rand = np.random.uniform(0,1,10000)

    samples = [f_inverse(r) for r in rand]

    plt.title("Accept Rejection Algorithm")

    plt.plot(x, y, label="f(x)")
    plt.hist(samples, bins=50,label='sampling', density=True)
    plt.legend()
    plt.savefig("assets/1_a.jpg")

if __name__ == "__main__":
    main()