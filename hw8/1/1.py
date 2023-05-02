import numpy as np
import random

def seed_everything():
    np.random.seed(311657007)
    random.seed(311657007)
seed_everything()

class YSimulator:
    
    def __init__(self, burnout, n_simulations, Idk_how_to_name_this_param, illness=0.01):
        self.burnout = burnout
        self.n_simulations = n_simulations
        self.Idk_how_to_name_this_param = Idk_how_to_name_this_param
        self.illness = illness

    def transition(self, y):

        size = y.shape[0]

        for _ in range(self.burnout + self.n_simulations * self.Idk_how_to_name_this_param):
            newy = y + np.random.rand() + np.random.rand(size) * self.illness
            newy = newy - np.floor(newy)
            newy = np.sort(newy)
            if np.all(np.diff(newy)>=0.1):
                y = newy
                yield newy
            else:
                yield y
    
    def simulate(self, burnin):
        history = burnin
        for newy in simulator.transition(burnin):
            history = np.row_stack([history, newy])
        return self.postprocessing(history[1:, ])
    
    def postprocessing(self, arr):
        arr = np.delete(arr, np.s_[:self.burnout], axis=0)
        arr = arr[np.arange(self.n_simulations) * self.Idk_how_to_name_this_param, :]
        return arr

size = 9
burnin = np.cumsum(np.repeat(1/size, size))

simulator = YSimulator(burnout=100, n_simulations=100, Idk_how_to_name_this_param=100)
res = simulator.simulate(burnin)
print(np.sum(np.diff(res)<=0.1))

