import numpy as np

np.random.seed(311657007)

def intensity_function(t):
    """Intensity function for non-homogeneous Poisson process"""
    return 3 + 4 / (t + 1)

def non_homogeneous_poisson(T=10):

    times = []
    params = [intensity_function(t) for t in np.linspace(0,10,1000)]
    upper = max(params)

    t, i = 0, 0
    while t < T:
        u = np.random.uniform(0,1,2)
        t = t - 1 / upper * np.log(u[0])
        if u[1] <= intensity_function(t) / upper and t < 10:
            times.append(t)
            i = i + 1
    return times



def main():
    times = non_homogeneous_poisson()
    print(times)
if __name__ == "__main__":
    main()