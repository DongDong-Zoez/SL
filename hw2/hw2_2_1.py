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
    return 1/4 + 2 * x ** 3 + 5 / 4 * x ** 4

def accept_rejection_algorithm(pdf, M):
    while True:
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if y*M <= pdf(x):
            return x
        
def app1():
    np.random.seed(311657007)
    M = 3.5
    x = np.linspace(0,1,100)
    y = pdf(x)

    plt.plot(x, [M for i in range(len(x))], '--',label='upper bound')

    samples = [accept_rejection_algorithm(pdf, M) for i in range(10000)]
    
    plt.subplot(1,2,1)
    plt.title("Accept Rejection Algorithm")

    plt.plot(x, y, label="f(x)")
    plt.hist(samples, bins=50,label='sampling', density=True)
    plt.legend()

###### Second application

def g1(x):
    return 1

def g2(x):
    return 4 * x ** 3

def g3(x):
    return 5 * x ** 4

def G1inv(x):
    return x

def G2inv(x):
    return x ** 0.25

def G3inv(x):
    return x ** 0.2

def pdf(x):
    p1 = 0.25
    p2 = 0.5
    p3 = 0.25
    return p1 * g1(x) + p2 * g2(x) + p3 * g3(x)

def composition_method(x,p1,p2):
    r = np.random.uniform(0,1)
    if x < p1:
        return G1inv(r)
    elif x < p2:
        return G2inv(r)
    else:
        return G3inv(r)

def app2():
    np.random.seed(311657007)
    p1 = 0.25
    p2 = 0.5
    rand = np.random.uniform(0,1,10000)
    x = np.linspace(0,1,100)
    y = pdf(x)

    samples = [composition_method(r, p1, p2) for r in rand]

    plt.subplot(1,2,2)

    plt.title("Composition Method")

    plt.plot(x, y, label="f(x)")
    plt.hist(samples, bins=50,label='sampling', density=True)
    plt.legend()
    plt.savefig("assets/2_1.jpg")

def main():
    app1()
    app2()

if __name__ == "__main__":
    main()