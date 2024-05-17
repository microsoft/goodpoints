'''
Standard thinning baseline algorithm.
'''

import numpy as np

from goodpoints.jax.rounding import stratified_round
from goodpoints.jax.reopt import reopt_simplex, reopt_cp
from goodpoints.jax.rounding import seq_to_w


def std_thin(kernel, points, w, rng_gen, *,
            inflate_size, out_size, coreset_type='integer'):
    '''
    Standard Thinning given weighted input points.

    Args:
        kernel: A broadcastable kernel function with mean-zero.
        points: A SliceablePoints instance containing input information.
        w: Weights, n-dimensional vector.
        rng_gen: Random number generator.
        inflate_size: Inflated size for rounding.
        out_size: Output size.
        coreset_type: {'integer', 'simplex', 'cp'}, type of weights.
    '''

    seq = stratified_round(w, inflate_size, rng_gen)
    inds = np.linspace(0, inflate_size-1, out_size, dtype=int)
    supp = seq[inds]
    w = seq_to_w(supp, points.length)
    if coreset_type == 'integer':
        return w, supp
    supp = np.unique(supp)
    if coreset_type == 'simplex':
        w_supp = reopt_simplex(kernel, points.subset(supp),
                               initval=w[supp])
    else:
        assert(coreset_type == 'cp')
        w_supp = reopt_cp(kernel, points.subset(supp))
    w = np.zeros_like(w)
    w[supp] = w_supp
    return w, supp
