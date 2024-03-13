import numpy as np

def linear_cl_scheduler(t, a, b):
    return min(1, 1.0 * a + (1 - a) * 1.0 * t / b)

def root_cl_scheduler(t, a, b):
    return min(1, np.sqrt(a**2 + (1 - a**2) * t / b))

def geometric_cl_scheduler(t, a, b):
    return min(1, 2**(np.log2(a) - np.log2(a * t / b)))


