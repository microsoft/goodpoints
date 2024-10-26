'''
Debias-then-compress (DTC) meta algorithm that unifies all debiased compression
methods from Li et al. 2024 and allows caching of intermediate results.

We also instantiate all six debiased compression algorithms individually in
this file with recommended parameters.
'''

import logging
import numpy as np

from goodpoints.jax.st import stein_thin, st_thin
from goodpoints.jax.lr_debias import lr_debias
from goodpoints.jax.noop_debias import noop_debias
from goodpoints.jax.kt import kt
from goodpoints.jax.compress import kt_compresspp
from goodpoints.jax.recomb_thin import recomb_thin
from goodpoints.jax.chol_thin import chol_thin
from goodpoints.jax.std_thin import std_thin
from goodpoints.jax.rounding import seq_to_w_supp, \
    log2_ceil, log4_ceil


def dtc(kernel, points, cache, *,
        debias_alg, compress_alg,
        debias_cfg, compress_cfg,
        debias_seed=1, compress_seed=1):
    '''
    Debias-then-compress meta algorithm.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance of length n.
        cache: A Cache instance.
        debias_alg: {'st', 'lr'}, debiasing algorithm.
        compress_alg:
            {'kt', 'kt_compresspp', 'recomb', 'chol', 'std', 'st'},
            compression algorithm.
        debias_cfg: Configuration for the debiasing algorithm.
        compress_cfg: Configuration for the compression algorithm.
        debias_seed: Random seed for the debiasing algorithm.
        compress_seed: Random seed for the compression algorithm.
    Returns:
        w_thin: (n,), weights of the coreset.
        supp: Support of the coreset.
    '''

    n = points.length
    rng_gen = cache.append_seed(debias_seed)
    def debias_exec_fn():
        if debias_alg == 'st':
            w, supp = stein_thin(kernel, points, **debias_cfg)
        elif debias_alg == 'lr':
            w, supp, _ = lr_debias(kernel, points,
                                   rng_gen=rng_gen, **debias_cfg)
        elif debias_alg == 'noop':
            w, supp = noop_debias(kernel, points)
        else:
            raise ValueError(f'Unknown debiasing algorithm: {debias_alg}')
        return {'w': w, 'supp': supp}

    logging.info(f'Debiasing with {debias_alg}...')
    debias_result = cache.blocking_advance(
        'debias',
        {'alg': debias_alg, **debias_cfg},
        debias_exec_fn
    )
    w_debias = debias_result['w'][:]

    rng_gen = cache.append_seed(compress_seed)
    def compress_exec_fn():
        if compress_alg == 'kt':
            coreset = kt(kernel, points, w_debias,
                         rng_gen=rng_gen, **compress_cfg)
        elif compress_alg == 'cpp':
            coreset = kt_compresspp(kernel, points, w_debias,
                                    rng_gen=rng_gen, **compress_cfg)
        elif compress_alg == 'recomb':
            w_thin, supp = recomb_thin(kernel, points, w_debias,
                                       rng_gen=rng_gen, **compress_cfg)
        elif compress_alg == 'chol':
            w_thin, supp = chol_thin(kernel, points, w_debias,
                                     rng_gen=rng_gen, **compress_cfg)
        elif compress_alg == 'std':
            w_thin, supp = std_thin(kernel, points, w_debias,
                                    rng_gen=rng_gen, **compress_cfg)
        elif compress_alg == 'st':
            w_thin, supp = st_thin(kernel, points, w_debias,
                                   rng_gen=rng_gen, **compress_cfg)
            supp = np.unique(supp) # remove duplicates
        else:
            raise ValueError(f'Unknown compression algorithm: {compress_alg}')

        if compress_alg in ['kt', 'cpp']:
            # These two compression algorithms only output the coreset with
            # multiplicity. We need to convert it to the weight vector and
            # support.
            w_thin, supp = seq_to_w_supp(coreset, n)
        return {'w': w_thin, 'supp': supp}

    logging.info(f'Compressing with {compress_alg}...')
    compress_result = cache.blocking_advance(
        'compress',
        {'alg': compress_alg, **compress_cfg},
        compress_exec_fn
    )
    w_thin, supp = compress_result['w'][:], compress_result['supp'][:]
    return w_thin, supp


def skt(kernel, points, out_size, seed=None):
    '''
    Stein Kernel Thinning (SKT) algorithm that produces an equal-weighted
    coreset.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance of length n.
        out_size: Size of the coreset.
        seed: Random seed.
    Returns:
        w_thin: (n,), weights of the coreset.
        supp: Support of the coreset.
    '''
    n = points.length
    rng_gen = np.random.default_rng(seed)
    inflate_size = out_size * (2 ** log2_ceil(n, out_size))
    w_debias, coreset = stein_thin(kernel, points, inflate_size)
    coreset = kt(kernel, points, w_debias, rng_gen,
                 inflate_size=inflate_size,
                 out_size=out_size)
    w_thin, supp = seq_to_w_supp(coreset, n)
    return w_thin, supp


def lskt(kernel, points, rank, seed=None):
    '''
    Low-rank Stein Kernel Thinning (LSKT) algorithm that produces an
    equal-weighted coreset. The output size is determined automatically
    to be within [sqrt(n), 2sqrt(n)).

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance of length n.
        rank: Rank of low-rank debiasing.
        seed: Random seed.
    Returns:
        w_thin: (n,), weights of the coreset.
        supp: Support of the coreset.
    '''
    n = points.length
    rng_gen = np.random.default_rng(seed)
    inflate_size = 4**log4_ceil(n)
    out_size = 2**log4_ceil(n)
    w_debias, supp, _ = lr_debias(kernel, points, rng_gen,
                                  rank=rank, num_iter=7*out_size)
    w_thin, supp = kt_compresspp(kernel, points, w_debias, rng_gen,
                                 inflate_size=inflate_size)
    return w_thin, supp


def sr(kernel, points, out_size, seed=None):
    '''
    Stein Recombination (SR) algorithm that produces a simplex-weighted coreset.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance of length n.
        out_size: Size of the coreset.
        seed: Random seed.
    Returns:
        w_thin: (n,), weights of the coreset.
        supp: Support of the coreset.
    '''
    n = points.length
    rng_gen = np.random.default_rng(seed)
    w_debias, _ = stein_thin(kernel, points, n)
    w_thin, supp = recomb_thin(kernel, points, w_debias, rng_gen,
                               out_size=out_size)
    return w_thin, supp


def lsr(kernel, points, out_size, seed=None):
    '''
    Low-rank Stein Recombination (SR) algorithm that produces a
    simplex-weighted coreset.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance of length n.
        out_size: Size of the coreset.
        seed: Random seed.
    Returns:
        w_thin: (n,), weights of the coreset.
        supp: Support of the coreset.
    '''
    n = points.length
    rng_gen = np.random.default_rng(seed)
    w_debias, supp, _ = lr_debias(kernel, points, rng_gen,
                                  rank=out_size, num_iter=7*out_size)
    w_thin, supp = recomb_thin(kernel, points, w_debias, rng_gen,
                               out_size=out_size)
    return w_thin, supp


def sc(kernel, points, out_size, seed=None):
    '''
    Stein Cholesky (SC) algorithm that produces a constant-preserving coreset.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance of length n.
        out_size: Size of the coreset.
        seed: Random seed.
    Returns:
        w_thin: (n,), weights of the coreset.
        supp: Support of the coreset.
    '''
    n = points.length
    rng_gen = np.random.default_rng(seed)
    w_debias, _ = stein_thin(kernel, points, n)
    w_thin, supp = chol_thin(kernel, points, w_debias, rng_gen,
                             out_size=out_size)
    return w_thin, supp


def lsc(kernel, points, out_size, seed=None):
    '''
    Low-rank Stein Cholesky (LSC) algorithm that produces a
    constant-preserving coreset.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance of length n.
        out_size: Size of the coreset.
        seed: Random seed.
    Returns:
        w_thin: (n,), weights of the coreset.
        supp: Support of the coreset.
    '''
    n = points.length
    rng_gen = np.random.default_rng(seed)
    w_debias, supp, _ = lr_debias(kernel, points, rng_gen,
                                  rank=out_size, num_iter=7*out_size)
    w_thin, supp = chol_thin(kernel, points, w_debias, rng_gen,
                             out_size=out_size)
    return w_thin, supp
