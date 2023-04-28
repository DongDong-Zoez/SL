import numpy as np
from utils import seed_everything, qSystem
import matplotlib.pyplot as plt

seed_everything()

def postprocessor(system):
    I = np.sum(np.apply_along_axis(lambda x: [x[i+1] - x[i] for i in range(len(x)-1)], 1, system["arrival_times"]), axis=1)
    S = np.sum(system["service_times"], axis=1)
    target = np.sum(system["spending_times"], axis=1)
    return target, S - I
q = qSystem(10, 2, 1, reduction="control_variate", proposal_sampler=postprocessor)
q.simulation(10000)
q.estimation()
q.pretty_print(q.ms, name="Sum")
q.pretty_print(q.cov)