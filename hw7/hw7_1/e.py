import numpy as np
from utils import seed_everything, qSystem
import matplotlib.pyplot as plt

seed_everything()

def postprocessor(system):

    N = system["num_customers_in_system"] + 1
    T = system["spending_times"]

    return T,N

q = qSystem(10, 2, 1, reduction="condition", proposal_sampler=postprocessor)
q.simulation(10000)
q.buffer[:,-1] = q.buffer[:,2] + 10
q.estimation()
q.pretty_print(q.ms, name="Sum")
q.pretty_print(q.cov)