'''
Dummy debiasing method that produces uniform weights.
'''


import numpy as np

def noop_debias(kernel, points):
    n = points.length
    w = np.ones(n) / n
    supp = np.arange(n)
    return w, supp
