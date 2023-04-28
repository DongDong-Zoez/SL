import numpy as np
import random
from prettytable import PrettyTable

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
        # assert target.shape == proposal.shape
        # smpl_size = target.shape[1]
        # for i in range(smpl_size):
        #     for j in range(i+1):
                
                # mask = (proposal[:, i] == j)
                # target[mask, i] = np.mean(target[mask, i])
        # return target.sum(axis=1)
        return proposal.sum(axis=1)

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


def seed_everything():
    np.random.seed(311657007)
    random.seed(311657007)

seed_everything()


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
