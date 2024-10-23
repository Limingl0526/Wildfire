import numpy as np

def stochastic_perturbation(x, epsilon=1e-10):
    U = np.random.uniform(epsilon, 1) 
    result = 1 + (np.log(U))**x
    return result
