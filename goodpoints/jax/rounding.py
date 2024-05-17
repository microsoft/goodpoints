'''
Rounding algorithms, including stratified resampling.
'''


import numpy as np


def stratified_round(w, m, rng_gen,
                     residual_first=True):
    '''
    Stratified resampling, with residual sampling first.

    Args:
        w: (n,) weights.
        m: int, number of output samples.
        rng_gen: Random number generator state.
        residual_first: bool, whether to do residual sampling first.
    Returns: (m,) indices of samples.
    '''
    n = w.size
    seq = []
    if residual_first:
        base = np.floor(w * m)
        for i in range(n):
            while base[i] > 0.5:
                seq.append(i)
                base[i] -= 1
        num_left = m - len(seq)
        if num_left == 0:
            return np.array(seq)
        w = (w * m - np.floor(w * m)) / num_left
        m = num_left
    seq = np.array(seq, dtype=int)

    # We use the default ordering to do stratified sampling because it is
    # assumed that nearby points are more similar so we want to put them into
    # the same bins.
    cumsum = np.cumsum(w)

    t = rng_gen.random([m]) # (m,)
    t = np.arange(m) / m + t / m # (m,)
    remain_seq = np.searchsorted(cumsum, t) # (m,)
    seq = np.concatenate([seq, remain_seq])
    seq = np.sort(seq)

    return seq


def seq_to_w(inds, n):
    w = np.zeros(n, dtype=float)
    for i in inds:
        w[i] += 1
    w /= len(inds)
    return w


def seq_to_w_supp(inds, n):
    supp = np.unique(inds)
    w = np.zeros(n, dtype=float)
    for i in inds:
        w[i] += 1
    w /= len(inds)
    return w, supp


def log2_ceil(n, m=1):
    '''
    Returns: ceil(log2(n/m))
    '''

    k = 0
    while m*(2**k) < n:
        k += 1
    return k


def log4_ceil(n, m=1):
    '''
    Returns: ceil(log4(n/m))
    '''

    k = 0
    while m*(4**k) < n:
        k += 1
    return k
