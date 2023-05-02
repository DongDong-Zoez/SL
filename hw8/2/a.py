import numpy as np
import random

def seed_everything():
    np.random.seed(311657007)
    random.seed(311657007)
seed_everything()


# Define the class
class GibbsSampler:
    def __init__(self, burnin, coefs):
        self.burnin = burnin
        self.coefs = coefs
        self.dim = len(burnin)

    def condition_1(self, burnin, d, threshold):
        Y = burnin.copy()
        bound = max(0, threshold - np.dot(burnin[self.coefs != self.coefs[d]], self.coefs[self.coefs != self.coefs[d]]))/self.coefs[d]
        Y[d] = np.random.exponential(scale=1) + bound
        return Y
    
    def condition_2(self, burnin, d, threshold):
        Y = burnin.copy()
        temp = threshold - np.dot(burnin[self.coefs != self.coefs[d]], self.coefs[self.coefs != self.coefs[d]])
        temp = 1 - np.exp(-temp)
        Y[d] = -np.log(1-np.random.uniform(0,1)*temp)
        return Y
    
    def simulation(self, threshold, total_n):
        Y = np.zeros((self.dim, total_n))
        Y[:,0] = self.burnin
        for i in range(1, total_n):
            for d in range(self.dim):
                if threshold > 8:
                    Y[:,i] = self.condition_1(Y[:,i-1], d, threshold)
                else:
                    Y[:,i] = self.condition_2(Y[:,i-1], d, threshold)
        return Y
    
    def extract_samples(self, simulation, burnout, Idk_how_to_name_this_param):
        Z = np.dot(self.coefs, simulation)
        Z = Z[burnout:]
        Z = Z[::Idk_how_to_name_this_param]
        return Z

# Define the parameters
burnout = 10100
n_simulations = 100*10**2
Idk_how_to_name_this_param = 100
coefs = np.array([1, 2, 3])
burnin = np.array([3, 3, 3])

gibbs_sampler = GibbsSampler(burnin, coefs)

threshold = 15
simulation = gibbs_sampler.simulation(threshold, burnout+n_simulations*Idk_how_to_name_this_param)
Z = gibbs_sampler.extract_samples(simulation, burnout, Idk_how_to_name_this_param)
print(np.mean(Z))

burnin = np.array([0.5, 0.5, 0.5])
gibbs_sampler = GibbsSampler(burnin, coefs)

threshold = 1
simulation = gibbs_sampler.simulation(threshold, burnout+n_simulations*Idk_how_to_name_this_param)
Z = gibbs_sampler.extract_samples(simulation, burnout, Idk_how_to_name_this_param)
print(np.mean(Z))
