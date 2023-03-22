import numpy as np
from typing import Callable, List, Optional, Tuple, Dict
from pprint import pprint
from prettytable import PrettyTable

from utils import (
    seed_everything, 
    trimmed_mean,  
    generate_multinormal_uniform,
)

seed_everything()

class Bootstrap:
    """
    Bootstrap class for estimating statistics and their confidence intervals
    using parametric or non-parametric bootstrap resampling methods.
    """
    
    def __init__(self, data: List[float]) -> None:
        """
        Constructor method for the Bootstrap class.
        
        Args:
            data (array-like): Input data for bootstrap resampling.
        """
        self.data = data
        self.table = PrettyTable()
        self.method = ""
        self.estimate_mean = 0.0
        self.estimate_std = 0.0
        self.CI = (0.0, 0.0)
        self.metadata = {}
        self.metadata["data"] = self.data
        
    def parametric(self, func: Callable, std: Optional[float] = None, 
                   n_resamples: int = 10000, conf_level: float = 0.95) -> Tuple[float, Tuple[float, float]]:
        """
        Parametric bootstrap method for estimating statistics and their confidence intervals.
        
        Args:
            func (callable): Function to apply to bootstrap resamples.
            n_resamples (int): Number of bootstrap resamples to generate (default: 1000).
            
        Returns:
            tuple: Tuple containing the estimated statistic and its confidence interval.
        """
        n = len(self.data)
        likelihood = np.mean(self.data)
        if std:
            params = generate_multinormal_uniform(np.array([likelihood]), np.array([[std]]), size=n_resamples)
        else:
            params = np.random.normal(loc=np.mean(self.data), scale=np.std(self.data), size=n_resamples)
        resamples = [func(np.random.normal(loc=p, scale=np.std(self.data), size=n)) for p in params]

        self.method = "Parametric (Normal)"
        self.estimate_std = self.boot_std(resamples)
        self.estimate_mean = np.mean(resamples)
        self.CI = np.percentile(resamples, [conf_level * 100 / 2, 100 - conf_level * 100 / 2])
        self.conf_level = conf_level

        self.metadata["Intro"] = "Assume data from population: %s with parameters (%s, %d)" %("N(μ, σ)", "unknown", std)

        self.recordHistory({
            "Method": self.method, 
            "Estimate B(Θ)": self.estimate_mean, 
            "Raw Estimate": np.mean(self.data), 
            "Estimate B(δ)": self.estimate_std, 
            "CI": self.CI,
            "α": self.conf_level,
        })
    
    def non_parametric(self, func: Callable, n_resamples: int = 10000, 
                       conf_level: float = 0.95) -> Tuple[float, Tuple[float, float]]:
        """
        Non-parametric bootstrap method for estimating statistics and their confidence intervals.
        
        Args:
            func (callable): Function to apply to bootstrap resamples.
            n_resamples (int): Number of bootstrap resamples to generate (default: 1000).
            
        Returns:
            tuple: Tuple containing the estimated statistic and its confidence interval.
        """
        n = len(self.data)
        resamples = [func(np.random.choice(self.data, size=n, replace=True)) for _ in range(n_resamples)]

        self.method = "Non Parametric"
        self.estimate_std = self.boot_std(resamples)
        self.estimate_mean = np.mean(resamples)
        self.CI = np.percentile(resamples, [conf_level * 100 / 2, 100 - conf_level * 100 / 2])
        self.conf_level = conf_level

        self.recordHistory({
            "Method": self.method, 
            "Estimate B(Θ)": self.estimate_mean, 
            "Raw Estimate": np.mean(self.data), 
            "Estimate B(δ)": self.estimate_std, 
            "CI": self.CI,
            "α": self.conf_level,
        })

    @staticmethod
    def boot_std(resamples: List[float]) -> float:
        """
        Calculate the bootstrap standard deviation of a set of resamples.
        
        Parameters:
            - self: the current object instance
            - resamples: an array-like object containing the resamples
            
        Returns:
            - The bootstrap standard deviation of the resamples.
        """
        bias = resamples - np.mean(resamples)  # Calculate the bias of the resamples.
        return 1 / (len(resamples) - 1) * sum(bias ** 2) ** 0.5  # Calculate and return the bootstrap standard deviation.
    
    def recordHistory(self, history: Dict[str, float]):
        """
        Records the history of the training process.

        Args:
            history (dict): A dictionary containing the history of the training process.
        """
        self.table.field_names = list(history.keys())
        self.table.add_row(list(history.values()))

    def displayResult(self):
        """
        Displays the result of the training process.
        """
        pprint(self.metadata)
        print(self.table)  # Prints the table containing the history of the training process.

    
def main():

    data = np.array([68, 78, 79, 83, 86, 88, 89, 91, 92, 97])
    std = 8
    n_boots = 1000
    boot = Bootstrap(data)
    boot.parametric(trimmed_mean, std, n_boots, conf_level=0.8)
    boot.non_parametric(trimmed_mean, n_boots, conf_level=0.8)
    boot.displayResult()

if __name__ == "__main__":
    main()