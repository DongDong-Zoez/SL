# SL HW7

In this assignment, we use markov chain and gibbs sampling to sample data.

## 1

We build an generator object for simulations conditional uniform dist. with less memory required 

### Algorithm

**Input:**

- ```burnout:``` relative to ```burnin``` (in the method of ```Ysimulator```), which indicate the number of samples we drop when ```burnin```
- ```n_simulations:``` number of simulations to simulate
- ```Idk_how_to_name_this_param:``` perodic sample data from simulations
- ```burnin:``` the start vector to generate, note that the vector is generated with diff > 0.1

**Output:**

An 2d array with shape (n_simlations, burnin_shape).

**Algorithm:**

condition: diff > 0.1

1. Set ```burnin``` (alias y) as the start vector (under condition)
2. Set ```newy``` = ```y``` + ```illness``` + x (x ~ Unif(0,1))
3. Yield ```newy``` AND Set ```newy``` = ```y``` IF ```newy``` statisfy condtion
4. ELSE Yield ```y``` 
5. Repeat 2. ~ ```n_simulations``` times and record as ```history```
6. Postprocessing ```history```

The algorithm ensure that each simulation will statisfy the condtion

We give a sample code to illustrate the whole process

```python
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
```

## 2

We build a Gibbs sampler object to simulate Y under two conditions

```python
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
```

The disadvantage of this code is that it cannot be widely applied to different conditions

### Algorithm

**Input:**

- ```burnout:``` relative to ```burnin``` (in the method of ```GibbsSampler```), which indicate the number of samples we drop when ```burnin```
- ```n_simulations:``` number of simulations to simulate
- ```Idk_how_to_name_this_param:``` perodic sample data from simulations
- ```burnin:``` the start vector to generate, note that the vector is generated with diff > 0.1
- ```coefs:``` the coefficient of the dist.

**Output:**

An 2d array with shape (n_simlations, burnin_shape).

**Algorithm:**

1. Set ```burnin``` (x_1, ..., x_n) as the start vector
2. For x_j in ```burnin``` Calc condition_pdf(x_1, ..., x_j-1, x_j+1, x_n) j = 1, ..., n
3. Repeat Step 2 ~ ```burnout``` + ```n_simulations``` times and record as ```history```
4. Set ```history``` = ```history``` * ```coefs```
5. Postprocessing ```history```

### Result

The estimation are the followings:

```
17.95543330079815
0.683692847647242
```