import numpy as np
import matplotlib.pyplot as plt

def box_muller(n):
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
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return z1, z2

def main():

    samples = box_muller(10000)
    
    plt.title("Box Muller Transformation")

    plt.hist(np.array(samples).flatten(), bins=50,label='sampling', density=True)
    plt.legend()

    plt.savefig("assets/3_1.jpg")

if __name__ == "__main__":
    main()
