"""Kernel herding.

Implementation of the kernel herding algorithm of 
  Yutian Chen, Max Welling, and Alex Smola.
  Super-Samples from Kernel Herding.
  https://arxiv.org/pdf/1203.3472.pdf
"""

import numpy as np

def herding(X, m, kernel, unique=False):
    """
    Returns herding coreset of size n/2^m as row indices into X
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      m: Thinning factor; output size is n/2^m
      kernel: Kernel function; kernel(y,X) returns array of kernel evaluations
        between y and each row of X
      unique: If True, constrains herding updates to never select the same
        row index more than once
    """
    n = X.shape[0]
    coreset_size = int(n/2**m)
    # print(X.shape, n, m, coreset_size)

    # Allocate memory for the coreset; important to assign dtype
    coreset = np.empty(coreset_size, dtype=int)

    # Initialize meanK vector with meanK[ii] = PnK(X[ii]) where Pn denotes
    # distibution of X
    meanK = np.empty(n)
    for ii in range(n):
        meanK[ii] = np.mean(kernel(X[ii, np.newaxis], X))
    
    # Our objective = PnK(x) - QK(x) where x denotes a candidate point in X, 
    # and Q denotes the coreset; and we are maximizing objective
    # since Q is initialized to 0, starting objective is simply meanK
    objective = meanK.copy()

    # At each step we add x_t = argmax_{x in X} PnK(x) - Q k(x) to Q, i.e.,
    # and update Q to (t+1) / (t+2) * Q   + 1 / (t+2) * dirac_{x_t} 
    # where we use t+1 because of Python indexing
    for t in range(coreset_size):
        # Add argmax of the objective to coreset
        coreset[t] =  np.argmax(objective)
        # x_{t} := X[coreset[t]]

        # We can write next step objective as
        # = PnK - Qnew K = PnK - (t+1) / (t+2) * Qk   + 1 / (t+2) * K(x_t, .)
        # = (t+1) / (t+2) * (PnK - Qk)    + 1 / (t+2) * ( PnK -  K(x_t, .) )
        # = (t+1) / (t+2) * objective + 1 / (t+2) * ( meanK - K(x_t, X) )
        # since we always consider points only in X
        objective = objective * (t+1) / (t+2) + (meanK - kernel( X[coreset[t], np.newaxis], X)) / (t+2)
        
        if unique:
            # If requested, ensure the same row index is not selected again
            objective[coreset[t]] = -np.inf
    return(coreset)
