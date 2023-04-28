import numpy as np
import random 

def seed_everything():
    np.random.seed(311657007)
    random.seed(311657007)
    
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
