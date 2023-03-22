# 統計學習第五次作業

在這次的作品中，我們手撕 bootstrap 的工程實現邏輯，除此之外，也實作了不同種類的 bootstrap (參數與非參)

Follow up the instruction, you may see the result

|        Method       |   Estimate B(Θ)   | Raw Estimate |    Estimate B(δ)     |             CI            |  α  |
|---------------------|-------------------|--------------|----------------------|---------------------------|-----|
| Parametric (Normal) | 85.08233909738001 |     85.1     | 0.03023678497435169  | [84.01273538 86.14437383] | 0.8 |
|    Non Parametric   | 85.10641666666666 |     85.1     | 0.032211036655777535 | [84.33333333 86.        ] | 0.8 |

## Realize Bootstrap sampling technique


```python
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
                   n_resamples: int = 1000, conf_level: float = 0.95) -> Tuple[float, Tuple[float, float]]:
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
    
    def non_parametric(self, func: Callable, n_resamples: int = 1000, 
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
        return sum(1 / (len(resamples) - 1) * bias ** 2) ** 0.5  # Calculate and return the bootstrap standard deviation.
    
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
```

This is a Python class Bootstrap that implements the parametric and non-parametric bootstrap methods for estimating statistics and their confidence intervals.

The Bootstrap class takes an array of data in its constructor, and has two main methods: `parametric()` and `non_parametric()`.

The `parametric()` method implements the parametric bootstrap method, where the data is assumed to come from a normal distribution. It takes a function func as an argument, which is applied to the bootstrap resamples. Additionally, the method takes an optional argument std which specifies the standard deviation of the normal distribution the data is assumed to come from. If std is not provided, the standard deviation is estimated from the data. The method returns a tuple containing the estimated statistic and its confidence interval.

The `non_parametric()` method implements the non-parametric bootstrap method, where the data distribution is estimated from the data itself. It takes a function func as an argument, which is applied to the bootstrap resamples. The method returns a tuple containing the estimated statistic and its confidence interval.

The Bootstrap class also has a method `displayResult()` that prints out the history of the training process, and a method recordHistory() that records the history of the training process.

 It generates some sample data, creates a Bootstrap object with the sample data, and applies the `parametric()` and `non_parametric()` methods to estimate the trimmed mean and its confidence interval. Finally, it displays the results.

 ## Usage

 ```sh
 chmod +x script.sh
 ./script.sh
 ```