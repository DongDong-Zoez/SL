import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)

def pdf(x):
    return x + 0.5

def accept_rejection_algorithm(pdf, M):
    while True:
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if y*M <= pdf(x):
            return x
        
def main():
    """
    PDF: .5 + x

    main idea:
        1. calculate f upper bound --> .5 + 1 = 1.5

    pseudo code:
        1. Random Sample U1, U2 from Uinf(0,1)
        2. IF M*U2 <= f(U1) do darwSample(U1)
        3. Repeat above step N times

    annotations:
        1. darwSample: a function to obtain sample
        2. M: upper bound of f
    """
    np.random.seed(311657007)
    M = 1.5
    x = np.linspace(0,1,100)
    y = pdf(x)

    plt.plot(x, [M for i in range(len(x))], '--',label='upper bound')

    samples = [accept_rejection_algorithm(pdf, M) for i in range(10000)]

    plt.title("Accept Rejection Algorithm")

    plt.plot(x, y, label="f(x)")
    plt.hist(samples, bins=50,label='sampling', density=True)
    plt.legend()
    plt.savefig("assets/1_b.jpg")

if __name__ == "__main__":
    main()