'''
Algorithms from Shetty et al. 2022 for near-linear-time equal-weighted
compression.
'''


import numpy as np
import math
import logging

from goodpoints.jax.kt import kernel_halve, \
    kernel_split, kernel_swap
from goodpoints.jax.rounding import stratified_round,\
    log2_ceil, log4_ceil


def kt_compresspp(kernel, points, w, rng_gen, *,
                  inflate_size,
                  delta=0.5,
                  g=4):
    '''
    KT-Compress++ algorithm from Li et al. 2024 targeting a mean-zero kernel
    and a weighted input.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance containing input information.
        w: Weights, n-dimensional vector.
        rng_gen: Random number generator state.
        inflate_size: Multiplicity, a power of 4.
        delta: Failure probability.
    Returns:
        A numpy array of indices of the coreset of length sqrt(inflate_size).
    '''

    log4_n = log4_ceil(inflate_size)
    assert(inflate_size == 4**log4_n)
    seq = stratified_round(w, inflate_size, rng_gen)
    logging.info(f'Compress++ with {seq.size} points...')
    coresets = compresspp(kernel, points.subset(seq), g, delta, rng_gen)
    coresets = [seq[coreset] for coreset in coresets]
    logging.info('KT-swap improvement...')
    coreset = kernel_swap(kernel, points, coresets,
                          rng_gen, mean_zero=True)
    return coreset


def compress(halve_fn, points, g, rng_gen,
             ordered=True, one_less_halve=False):
    '''
    Compress from Shetty et al. 2022.

    Args:
        halve_fn: Halving function.
        points: A SliceablePoints instance containing input information.
        g: Oversampling parameter.
        rng_gen: Random number generator state.
        ordered: Whether to return the coreset in the same order as the input.
        one_less_halve:
            Whether to skip the final halving and return a size 2^{g+1} sqrt(n)
            coreset.
    Returns: A numpy array representing the final coreset of size 2^g sqrt(n).
    '''
    n = points.length
    if n == 4**g:
        return np.arange(n)

    if ordered:
        perm = np.arange(n)
    else:
        perm = rng_gen.permutation(n)

    n_prime = n // 4
    S = []
    for i in range(4):
        inds = perm[i*n_prime:(i+1)*n_prime]
        S_prime = compress(halve_fn, points.subset(inds), g,
                           rng_gen, one_less_halve=False)
        S.append(inds[S_prime])
    S_cat = np.concatenate(S)
    if ordered:
        S_cat = np.sort(S_cat)
    if one_less_halve:
        return S_cat
    S_halve = halve_fn(points.subset(S_cat), rng_gen)
    return S_cat[S_halve]


def compresspp(kernel, points, g, delta, rng_gen,
                one_less_halve=True):
    '''
    Compress++ from Shetty et al. 2022 specialized to kernel thinning.
    Thins n inputs points to m points.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance containing input information.
        g: Inflation parameter so that compress results in 2^g sqrt(n) points.
        delta: Failure probabilities, delta_i = delta/n.
        rng_gen: Random number generator state.
        one_less_halve:
            Whether to skip the final halving in compress and instead thin from
            2^{g+1} sqrt(n) to sqrt(n) with 2^{g+1} coresets.

    Returns:
        2^g corests of size m.
    '''

    n = points.length
    log2_n = log2_ceil(n)
    assert(n == 2**log2_n)
    assert(log2_n % 2 == 0)

    if g >= log2_n // 2:
        logging.info(f'g ({g}) is too large for n ({n})! Setting g to '
                     f'{log2_n // 2 - 1}.')
        g = log2_n // 2 - 1

    # Compute deltas for halving and thinning.
    beta_n = log2_n // 2 - (g + 1)
    halve_delta0 = delta / (4 * n * (2**g) * (g + (beta_n+1)*(2**g)))
    kt_delta = g / (g + (beta_n+1)*(2**g)) * delta

    def halve_fn(points, rng_gen):
        # The delta to halve is always proportional to input size squared.
        n = points.length
        halve_delta = (n ** 2) * halve_delta0
        S0, S1 = kernel_halve(kernel, points, delta, rng_gen)
        if rng_gen.random() < 0.5:
            return S0
        else:
            return S1
    logging.info('Compressing to 2^g sqrt(n) points...')
    S_c = compress(halve_fn, points, g, rng_gen,
                   one_less_halve=one_less_halve) # size: 2^g sqrt(n)
    assert((S_c.shape[0] ==
            (2 if one_less_halve else 1) * 2**g * (2**(log2_n//2))))

    logging.info('Kernel thinning to sqrt(n) points...')
    coresets_cpp = kernel_split(kernel, points.subset(S_c),
                                g+(1 if one_less_halve else 0),
                                kt_delta,
                                rng_gen)
    coresets_cpp = [S_c[coreset] for coreset in coresets_cpp]
    return coresets_cpp
