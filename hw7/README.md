# SL HW7

In this assignment, we use variance reduction techniques to simulate a queueing system.

## 1. Utils

We create a queueing system (qSystem) that inherits from the VarianceReduction object.

```python
class VarianceReduction:

    def __init__(self, func, n_samples, method, *args, **kwargs):
        self.func = func
        self.n_samples = n_samples
        self.method = method
        self.transform = lambda x: x

        self.args = args
        self.kwargs = kwargs

    def __monte_carlo(self, func):
        return func

    def __antithetic(self, func):
        return lambda x: (func(x) + func(1 - x)) / 2

    def __control_variate(self, target, proposal):
        mu = np.mean(proposal)
        c = - np.cov(target, proposal)[0, 1] / np.var(proposal)
        return target + c * (proposal - mu)
    
    def __condition(self, target, proposal):
        assert target.shape == proposal.shape
        # num_smpls = target.shape[0]
        smpl_size = target.shape[1]
        # prob_proposal = np.zeros((smpl_size, smpl_size))
        # cond_target = np.zeros((smpl_size, smpl_size))
        for i in range(smpl_size):
            for j in range(i+1):
                # mask = (proposal[:,i] == j)
                # cond_target[i,j] += np.mean(target[mask, i])
                # prob_proposal[i,j] += np.mean(proposal[:, i] == j)

                mask = (proposal[:, i] == j)
                target[mask, i] = np.mean(target[mask, i])
        # print(target)
        # for i in range(num_smpls):
        #     for j in range(smpl_size):
        #         target[i, j] = cond_target[i,j] * prob_proposal[j, proposal[i, j]]
        return target.sum(axis=1)

    def preprocessor(self, func):
        if self.method == "antithetic":
            return self.__antithetic(func)
        else:
            return self.__monte_carlo(func)
        
    def postprocessor(self, data):
        try:
            proposal_sampler = self.kwargs["proposal_sampler"]
        except KeyError as e:
            raise e
        if self.method == "control_variate":
            target, proposal = proposal_sampler(data)
            return self.__control_variate(target, proposal)
        elif self.method == "condition":
            target, proposal = proposal_sampler(data)
            return self.__condition(target, proposal)
        else:
            return None

class Clock:
    def __init__(self):
        self.t = 0

    def walk(self, times):
        self.t += times

    def reset(self):
        self.t = 0


class qSystem(VarianceReduction):
    def __init__(
        self,
        num_customers,
        customer_per_minute,
        service_per_customer,
        reduction=None,
        proposal_sampler=None,
        *args,
        **kwargs,
    ):

        self.num_customers = num_customers
        self.customer_per_minute = customer_per_minute
        self.service_per_customer = service_per_customer
        self.reduction = reduction

        super().__init__(
            qSystem.generate_arrival_time(1/self.customer_per_minute),
            self.num_customers,
            self.reduction,
            proposal_sampler=proposal_sampler if proposal_sampler is not None else qSystem.generate_service_time(
                1/self.service_per_customer),
            *args,
            **kwargs,
        )

        self.clock = Clock()
        self.service_busy_until = 0
        self.buffer = []

        self.system = {
            "arrival_times":  [],
            "service_times":  [],
            "num_customers_in_system": [],
            "queueing_times": [],
            "starting_times": [],
            "ending_times": [],
            "spending_times": [],
        }

    @staticmethod
    def generate_arrival_time(lam=0.5):
        return lambda x: -lam * np.log(x)

    @staticmethod
    def generate_service_time(lam=1):
        return lambda x: -lam * np.log(x)
    
    def __get_uniform(self):
        self.rand = np.random.uniform(0, 1, self.num_customers)

    def event(self):
        self.__get_uniform()
        interarrival_times = self.preprocessor(qSystem.generate_arrival_time())(self.rand)
        service_times = self.preprocessor(qSystem.generate_service_time())(self.rand)

        num_customers_in_system = -1
        temp = []
        for interarrival_time, service_time in zip(interarrival_times, service_times):
            num_customers_in_system += 1
            self.clock.walk(interarrival_time)

            arrival_time = self.clock.t
            service_start_time = max(arrival_time, self.service_busy_until)
            service_end_time = service_start_time + service_time
            temp.append(service_end_time)
            for i in range(len(temp)):
                if arrival_time > temp[i]:       
                    num_customers_in_system -= 1
                else:
                    temp = temp[i:]
                    break
            queueing_time = service_start_time - arrival_time
            spending_time = service_end_time - arrival_time

            self.service_busy_until = service_end_time

            

            self.recordEvent(arrival_time, service_time, num_customers_in_system, service_start_time,
                             service_end_time, queueing_time, spending_time)
            
        self.customers_in_system = self.system

    def recordEvent(self, arrival_time, service_time, num_customers_in_system, service_start_time, service_end_time, queueing_time, spending_time):
        self.system["arrival_times"].append(arrival_time)
        self.system["service_times"].append(service_time)
        self.system["num_customers_in_system"].append(num_customers_in_system)
        self.system["starting_times"].append(service_start_time)
        self.system["ending_times"].append(service_end_time)
        self.system["queueing_times"].append(queueing_time)
        self.system["spending_times"].append(spending_time)

    def sampler(self):
        self.event()
        self.clock.reset()
        self.service_busy_until = 0

    def postprocessing(self):
        self.buffer = []
        for k, v in self.system.items():
            v = np.array(v).reshape(-1, self.num_customers)
            self.system[k] = v
            v = np.apply_along_axis(np.sum, 1, v)
            self.buffer = self.buffer + v.tolist()
        self.buffer = np.array(self.buffer).reshape(-1, len(self.system), order="F")
        transformed = self.postprocessor(self.system)
        if transformed is not None:
            self.buffer[:,-1] = transformed

    def simulation(self, n_sample):
        self.reset()
        for _ in range(n_sample):
            self.sampler()
        self.clock.reset()
        self.postprocessing()

    def estimation(self):
        ms = np.mean(self.buffer, axis=0)
        cov = np.cov(self.buffer.T)
        self.reset()
        self.registry("cov", cov)
        self.registry("ms", ms)

    def registry(self, k, v):
        setattr(self, k, v)

    def pretty_print(self, arr, name="Covariance"):
        if len(arr.shape) < 2:
            arr = arr[None, ...]
        table = PrettyTable()
        keys = list(self.system.keys())
        table.field_names = [name] + keys
        for i in range(arr.shape[0]):
            table.add_row([keys[i]] + list(np.round(arr[i,:], 2)))
        print(table)

    def reset(self):

        self.arrival_times = []
        self.service_times = []
        self.queueing_times = []
        self.starting_times = []
        self.ending_times = []
        self.spending_times = []

        self.service_busy_until = 0
```

### Raw Estimation

To simulate the queueing system with raw estimator, we can write

```python
q = qSystem(10, 2, 1)
q.simulation(10000)
q.estimation()
q.pretty_print(q.ms, name="Sum")
q.pretty_print(q.cov)
```

where ```qSystem(10, 2, 1)``` stands for 10 customers and 2 customers per minute and service time for the single queueing system is 1 customers per minute.

We then simulate 10000 samples with sample size 10 (10 customers) to construct the queueing system

The result is shown as following:

| Sum                     | arrival_times | service_times | num_customers_in_system | queueing_times | starting_times | ending_times | spending_times |
|-------------------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
| arrival_times           | 27.43         | 9.98          | 27.09                   | 24.85          | 52.29          | 62.27        | 34.83          |


| Covariance              | arrival_times | service_times | num_customers_in_system | queueing_times | starting_times | ending_times | spending_times |
|-------------------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
| arrival_times           | 95.22         | 27.25         | 22.08                   | 100.41         | 195.63         | 222.88       | 127.66         |
| service_times           | 27.25         | 9.94          | -0.14                   | 24.61          | 51.86          | 61.8         | 34.55          |
| num_customers_in_system | 22.08         | -0.14         | 60.02                   | 47.86          | 69.94          | 69.81        | 47.72          |
| queueing_times          | 100.41        | 24.61         | 47.86                   | 125.14         | 225.55         | 250.15       | 149.74         |
| starting_times          | 195.63        | 51.86         | 69.94                   | 225.55         | 421.18         | 473.04       | 277.4          |
| ending_times            | 222.88        | 61.8          | 69.81                   | 250.15         | 473.04         | 534.83       | 311.95         |
| spending_times          | 127.66        | 34.55         | 47.72                   | 149.74         | 277.4          | 311.95       | 184.29         |


### Antithetic

To simulate the queueing system with antithetic variables, we can write

```python
q = qSystem(10, 2, 1, reduction="antithetic")
q.simulation(10000)
q.estimation()
q.pretty_print(q.ms, name="Sum")
q.pretty_print(q.cov)
```

The result is shown as following:

|---------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
|      Sum      | arrival_times | service_times | num_customers_in_system | queueing_times | starting_times | ending_times | spending_times |
|---------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
| arrival_times |     27.47     |      9.99     |          26.91          |     22.64      |     50.11      |     60.1     |     32.63      |

|        Covariance       | arrival_times | service_times | num_customers_in_system | queueing_times | starting_times | ending_times | spending_times |
|-------------------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
|      arrival_times      |     17.05     |      4.84     |           4.8           |     17.61      |     34.65      |    39.49     |     22.45      |
|      service_times      |      4.84     |      1.76     |          -0.04          |      4.18      |      9.02      |    10.78     |      5.94      |
| num_customers_in_system |      4.8      |     -0.04     |          15.36          |     11.86      |     16.66      |    16.61     |     11.81      |
|      queueing_times     |     17.61     |      4.18     |          11.86          |     22.44      |     40.05      |    44.23     |     26.62      |
|      starting_times     |     34.65     |      9.02     |          16.66          |     40.05      |      74.7      |    83.72     |     49.07      |
|       ending_times      |     39.49     |     10.78     |          16.61          |     44.23      |     83.72      |     94.5     |     55.01      |
|      spending_times     |     22.45     |      5.94     |          11.81          |     26.62      |     49.07      |    55.01     |     32.56      |

### Control variate S

To simulate the queueing system with control variate S, we can write

```python
def postprocessor(system):
    S = np.sum(system["service_times"], axis=1)
    target = np.sum(system["spending_times"], axis=1)
    return target, S
q = qSystem(10, 2, 1, reduction="control_variate", proposal_sampler=postprocessor)
q.simulation(10000)
q.estimation()
q.pretty_print(q.ms, name="Sum")
q.pretty_print(q.cov)
```

The result is shown as following:


|      Sum      | arrival_times | service_times | num_customers_in_system | queueing_times | starting_times | ending_times | spending_times |
|---------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
| arrival_times |     27.43     |      9.98     |          27.09          |     24.85      |     52.29      |    62.27     |     34.83      |


|        Covariance       | arrival_times | service_times | num_customers_in_system | queueing_times | starting_times | ending_times | spending_times |
|-------------------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
|      arrival_times      |     95.22     |     27.25     |          22.08          |     100.41     |     195.63     |    222.88    |     32.98      |
|      service_times      |     27.25     |      9.94     |          -0.14          |     24.61      |     51.86      |     61.8     |      -0.0      |
| num_customers_in_system |     22.08     |     -0.14     |          60.02          |     47.86      |     69.94      |    69.81     |      48.2      |
|      queueing_times     |     100.41    |     24.61     |          47.86          |     125.14     |     225.55     |    250.15    |     64.24      |
|      starting_times     |     195.63    |     51.86     |          69.94          |     225.55     |     421.18     |    473.04    |     97.22      |
|       ending_times      |     222.88    |      61.8     |          69.81          |     250.15     |     473.04     |    534.83    |     97.22      |
|      spending_times     |     32.98     |      -0.0     |           48.2          |     64.24      |     97.22      |    97.22     |     64.25      |

### Antithetic

To simulate the queueing system with control variate S - I, we can write

```python
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
```

The result is shown as following:

|      Sum      | arrival_times | service_times | num_customers_in_system | queueing_times | starting_times | ending_times | spending_times |
|---------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
| arrival_times |     27.43     |      9.98     |          27.09          |     24.85      |     52.29      |    62.27     |     34.83      |

|        Covariance       | arrival_times | service_times | num_customers_in_system | queueing_times | starting_times | ending_times | spending_times |
|-------------------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
|      arrival_times      |     95.22     |     27.25     |          22.08          |     100.41     |     195.63     |    222.88    |     20.87      |
|      service_times      |     27.25     |      9.94     |          -0.14          |     24.61      |     51.86      |     61.8     |     -1.72      |
| num_customers_in_system |     22.08     |     -0.14     |          60.02          |     47.86      |     69.94      |    69.81     |     30.45      |
|      queueing_times     |     100.41    |     24.61     |          47.86          |     125.14     |     225.55     |    250.15    |      43.9      |
|      starting_times     |     195.63    |     51.86     |          69.94          |     225.55     |     421.18     |    473.04    |     64.77      |
|       ending_times      |     222.88    |      61.8     |          69.81          |     250.15     |     473.04     |    534.83    |     63.05      |
|      spending_times     |     20.87     |     -1.72     |          30.45          |      43.9      |     64.77      |    63.05     |      42.2      |

### Condition

To simulate the queueing system with condition on N, we can write

```python
def postprocessor(system):

    N = system["num_customers_in_system"]
    T = system["spending_times"]

    return T,N

q = qSystem(10, 2, 1, reduction="condition", proposal_sampler=postprocessor)
q.simulation(10000)
q.estimation()
q.pretty_print(q.ms, name="Sum")
q.pretty_print(q.cov)
```

The result is shown as following:

|      Sum      | arrival_times | service_times | num_customers_in_system | queueing_times | starting_times | ending_times | spending_times |
|---------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
| arrival_times |     27.43     |      9.98     |          27.09          |     24.85      |     52.29      |    62.27     |     37.09      |

|        Covariance       | arrival_times | service_times | num_customers_in_system | queueing_times | starting_times | ending_times | spending_times |
|-------------------------|---------------|---------------|-------------------------|----------------|----------------|--------------|----------------|
|      arrival_times      |     95.22     |     27.25     |          22.08          |     100.41     |     195.63     |    222.88    |     22.08      |
|      service_times      |     27.25     |      9.94     |          -0.14          |     24.61      |     51.86      |     61.8     |     -0.14      |
| num_customers_in_system |     22.08     |     -0.14     |          60.02          |     47.86      |     69.94      |    69.81     |     60.02      |
|      queueing_times     |     100.41    |     24.61     |          47.86          |     125.14     |     225.55     |    250.15    |     47.86      |
|      starting_times     |     195.63    |     51.86     |          69.94          |     225.55     |     421.18     |    473.04    |     69.94      |
|       ending_times      |     222.88    |      61.8     |          69.81          |     250.15     |     473.04     |    534.83    |     69.81      |
|      spending_times     |     22.08     |     -0.14     |          60.02          |     47.86      |     69.94      |    69.81     |     60.02      |

## 2. Important Sampling

To simulate the probabilities of S_N < -5, we can simply set the reponse of the function to binary and take the mean of the monte carlo simulation to approximate the population.

```python
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

n_simulations = 1000000

N = [generated_N() for _ in range(n_simulations)]
```

The result is shownn as following:

```python
1e-5
```

But the approximation is not proper for the case, since S_N < -5 is hard to achieve, in my experimental, only 10 cases out of 1000000 statisfy the statement, therefore, we can sample from another distribution and times the likelihood of the event occur

```python
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
```

The result is shown as following:

```python
1.449864581568471e-05
```

which is close to the simulations of monte carlo method but only take 10000 samples points.