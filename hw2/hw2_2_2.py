import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings 
warnings.filterwarnings("ignore")

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)

###### First application

def pdf(x):
    return np.exp(-x) + 2 * np.exp(-2 * x) - 3 * np.exp(-3 * x)

def accept_rejection_algorithm(pdf, M):
    while True:
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if y*M <= pdf(x):
            return x
        
def app1():
    np.random.seed(311657007)
    M = 0.673845
    x = np.linspace(0,1,100)
    
    # we add .5465 since the area under curve of x belong to [0,1] is about 0.5465
    y = pdf(x) + 0.5465

    plt.plot(x, [M for i in range(len(x))], '--',label='upper bound')

    samples = [accept_rejection_algorithm(pdf, M) for i in range(1000000)]

    plt.subplot(1,2,1)
    plt.title("Accept Rejection Algorithm")

    plt.plot(x, y, label="f(x)")
    plt.hist(samples, bins=50,label='sampling', density=True)
    plt.legend()

###### Second application

import numpy as np
from scipy.optimize import brentq

def pdf(x):
    return np.exp(-x) + 2 * np.exp(-2*x) - 3 * np.exp(-3*x) 

def target(x):
    return 1 - np.exp(-x) - np.exp(-2*x) + np.exp(-3*x)

def cdf(x):
    return (target(x) - target(0)) / (target(1) - target(0))

def inverse_cdf(y):
    return brentq(lambda x: cdf(x) - y, a=0, b=1)

def app2():
    np.random.seed(311657007)
    rand = np.random.uniform(0,1,1000000)
    x = np.linspace(0,1,100)
    # we add .5465 since the area under curve of x belong to [0,1] is about 0.5465
    y = pdf(x) + 0.5465

    samples = [inverse_cdf(r) for r in rand]

    plt.subplot(1,2,2)

    plt.title("Inverse Transforms")

    plt.plot(x, y, label="f(x)")
    plt.hist(samples, bins=50,label='sampling', density=True)
    plt.legend()
    plt.savefig("assets/2_2.jpg")

def main():
    app1()
    app2()

if __name__ == "__main__":
    main()