import numpy as np
from utils import seed_everything, qSystem
import matplotlib.pyplot as plt

seed_everything()

q = qSystem(10, 2, 1, reduction="antithetic")
q.simulation(10000)
q.estimation()
q.pretty_print(q.ms, name="Sum")
q.pretty_print(q.cov)