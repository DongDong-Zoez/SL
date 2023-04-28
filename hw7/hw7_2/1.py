from utils import seed_everything
import numpy as np

def generated_N(mu=1, sigma=1, size=1):
    curr = 0
    while True:
        x = np.random.normal(mu,sigma,size)
        curr += x
        if curr < -5:
            return 1
        elif curr > 5:
            return 0

seed_everything()

n = 50
n_simulations = 1000000

N = [generated_N() for _ in range(n_simulations)]
print(np.mean(N))