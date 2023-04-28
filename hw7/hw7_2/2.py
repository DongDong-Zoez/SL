import numpy as np
from scipy import stats
from utils import seed_everything

seed_everything()

def generated_N(mu, sigma, p_x, q_x):
    curr = 0
    prob, qrob = 1, 1
    while True:
        x = np.random.normal(mu,sigma,1)
        prob *= p_x.pdf(x)
        qrob *= q_x.pdf(x)
        curr += x
        if curr < -5:
            return prob / qrob
        elif curr > 5:
            return 0
        
f_x = lambda mu, sigma: generated_N(mu, sigma)

def distribution(mu=0, sigma=1):
    # return probability given a value
    distribution = stats.norm(mu, sigma)
    return distribution

num_smpls = 10000


p_x = distribution(1, 1)
q_x = distribution(-1, 1)

value_list = []
for i in range(num_smpls):

    value = generated_N(-1, 1, p_x, q_x)

    value_list.append(value)

print(np.mean(value_list))