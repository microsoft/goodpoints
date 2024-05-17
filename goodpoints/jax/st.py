'''
Stein Thinning algorithm from Riabiz et al. 2021 extended to various
weight types.
'''


import numpy as np

from goodpoints.jax.kt_swap_ls import kernel_swap_ls
from goodpoints.jax.rounding import stratified_round
from goodpoints.jax.reopt import reopt_simplex, reopt_cp


def st_thin(kernel, points, w, rng_gen, *,
            inflate_size, out_size, coreset_type='integer'):
    '''
    Stein Thinning given weighted input points.

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
    w, supp = stein_thin(kernel, points.subset(seq), out_size,
                         weight_type='integer')
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


def stein_thin(kernel, points, out_size, weight_type='integer'):
    '''
    Stein Thinning algorithm.

    Args:
        kernel: A broadcastable kernel function with mean-zero.
        points: A SliceablePoints instance containing input information.
        out_size: int, output support size.
        weight_type: {'integer', 'simplex', 'cp'}, type of weights.

    Returns:
        w: (n,), weight vector.
        supp:
            (m,), indices of the support of w. If weight_type == 'integer',
            then supp is the indices of the coreset (allowing repeats).
    '''

    n = points.length

    # Initialize the weight vector to put 1 at the point with the smallest
    # diagonal kernel value.
    K_diag = kernel(points, points)
    k = K_diag.argmin().item()
    w = np.zeros(n)
    w[k] = 1.0
    supp = np.array([k])

    # Then greedily add in points to maximally reduce the MMD.
    w, supp = kernel_swap_ls(kernel, points, w, supp,
                             weight_type,
                             grow_until=out_size)

    return w, supp
