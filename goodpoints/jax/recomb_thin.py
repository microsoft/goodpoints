'''
Recombination Thinning algorithm from Li et al. (2024) that produces a
simplex-weighted coreset given weighted points.
'''

import math
import logging
import numpy as np
import jax
import jax.numpy as jnp

from goodpoints.jax.rpc import weighted_rpcholesky
from goodpoints.jax.reopt import reopt_simplex
from goodpoints.jax.kt_swap_ls import kernel_swap_ls


def recomb_thin(kernel, points, w, rng_gen, *,
                out_size):
    '''
    Recombination Thinning.

    Args:
        kernel: A broadcastable kernel function with mean-zero.
        points: A SliceablePoints instance with input information.
        w: Weights, n-dimensional vector.
        rng_gen: Random number generator.
        out_size: Output size.
    '''
    logging.info(f'Running rpcholesky to get a low-rank matrix'
                 ' for recombination.')
    F, _ = weighted_rpcholesky(kernel, points, w, out_size - 1, rng_gen)
    w_sqrt = np.sqrt(w)
    A = np.array(F).T # (k, n)

    # Add constraint.
    A = np.concatenate([A, w_sqrt[None]], 0)
    logging.info(f'Finding basis feasible solution...')
    w_prime_sqrt, supp = compute_bfs_fast(A, w_sqrt) # |supp| = k
    w = w_prime_sqrt * w_sqrt
    logging.info(f'Recombination yields support size {supp.size}.')

    logging.info('KT-swap improvement...')
    w, supp = kernel_swap_ls(kernel, points, w, supp, 'simplex')

    logging.info('Finally compute the optimal simplex weight.')
    w_supp = reopt_simplex(kernel, points.subset(supp),
                           initval=w[supp])
    w = np.zeros_like(w)
    w[supp] = w_supp

    return w, supp


def compute_bfs(A, x0, eps=1e-8):
    '''
    Given nxm matrix A, a vector x0 such that Ax0 = b with x0 in the simplex,
    find a basic feasible solution x such that Ax = b with x in the simplex and
    that it has at most n non-zero entries. This takes O(n^2 m).

    Args:
        A: nxm matrix.
        x0: m-dimensional vector.
        eps: Tolerance for numerical issues.
    '''
    n, m = A.shape

    _, _, Vh = np.linalg.svd(A)
    V = Vh[-(m-n):]

    x = x0

    is_zero = np.zeros(m, dtype=bool)
    for i in range(m-n):
        v = V[i]
        k = np.argmin(np.where(v > eps, x / np.where(v > eps, v, 1.),
                               np.inf))
        x = x - (x[k] / v[k]) * v
        V = V - (V[:, k] / v[k])[:, None] * v

        # Manually zero out x[k] and V[:, k] to avoid numerical issues.
        x[k] = 0
        V[:, k] = 0
        is_zero[k] = True

    inds = np.arange(m)[np.logical_not(is_zero)]

    return x, inds


def compute_bfs_fast(A, x0, eps=1e-8):
    '''
    Divide-and-conquer algorithm for computing a basic feasible solution
    by Tchernychova 2015. This takes O(n^3 log m).

    Arguments are the same as compute_bfs.
    '''
    n, m = A.shape
    x0 = np.copy(x0) # will modify x0 in place

    # inds will keep track of non-zero entries of x0.
    inds = np.arange(m)
    inds = inds[x0 > 0]
    N = len(inds)
    while N > 2 * n:
        # Divide A into 2n blocks.
        A_list = []
        h = int(math.floor(N / (2 * n))) # size of each block
        cur = 0
        while True:
            nxt = min(cur + h, N)
            Ai = A[:, inds[cur:nxt]] @ x0[inds[cur:nxt]]
            cur = nxt
            A_list.append(Ai)
            if nxt == N:
                break

        Ac = np.stack(A_list, axis=1) # nx2n or nx(2n+1)
        xc = np.ones(Ac.shape[-1], dtype=Ac.dtype) # (2n,) or (2n+1,)
        xc_bfs, inds_bfs = compute_bfs(Ac, xc, eps)
        mask_bfs = np.zeros_like(xc_bfs, dtype=bool)
        mask_bfs[inds_bfs] = True # indices that are in the support

        new_inds = []
        for i in range(len(xc_bfs)):
            I = inds[i*h:min((i+1)*h, N)]
            if mask_bfs[i] and abs(xc_bfs[i]) > eps:
                new_inds.append(I)
                x0[I] = x0[I] * xc_bfs[i]
            else:
                x0[I] = 0
        x0 = np.maximum(x0, 0) # stability
        inds = np.concatenate(new_inds)
        N = len(inds)

    if len(inds) >= n + 1:
        x_out = np.zeros([m])
        x_out[inds], inds_out = compute_bfs(A[:, inds], x0[inds], eps)
        inds = inds[inds_out]
    else:
        x_out = x0
    return x_out, inds
