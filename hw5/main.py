import argparse
import numpy as np
from utils import trimmed_mean
from .hw5 import Bootstrap

def argparser() -> argparse.Namespace:
    """
    Function that defines and parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Bootstrap estimator")
    parser.add_argument("-m", "--method", choices=["parametric", "non-parametric"],
                        default=["parametric", "non-parametric"], help="Bootstrap method to use.")
    parser.add_argument("-f", "--function", choices=["trimmed_mean", "median"],
                        default="trimmed_mean", help="Function to estimate.")
    parser.add_argument("-s", "--std", type=float, default=8,
                        help="Standard deviation for parametric bootstrap method.")
    parser.add_argument("-n", "--n_resamples", type=int, default=10000,
                        help="Number of bootstrap resamples to generate.")
    parser.add_argument("-c", "--conf_level", type=float, default=0.8,
                        help="Confidence level of the bootstrap interval.")
    parser.add_argument("-d", "--data", type=float, nargs="+",
                        default=[68, 78, 79, 83, 86, 88, 89, 91, 92, 97],
                        help="Data to use for bootstrap resampling.")
    return parser.parse_args()

def main():
    
    args = argparser()
    data = np.array(args.data)
    boot = Bootstrap(data)
    if args.function:
        func = eval(args.function)
    else:
        func = np.mean
    for m in args.method:
        if m == "parametric":
            boot.parametric(func, args.std, args.n_resamples, conf_level=args.conf_level)
        elif m == "non-parametric":
            boot.non_parametric(func, args.n_resamples, conf_level=args.conf_level)
    boot.displayResult()

if __name__ == "__main__":
    main()
