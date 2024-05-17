'''
Randomly pivoted Cholesky algorithm from Chen et al. (2023).
'''

import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from tqdm import tqdm

from goodpoints.jax.pd import nearestPD


def weighted_rpcholesky(kernel, points, w, rank, rng_gen):
    def K_col_fn(j, aux):
        points, w = aux
        return kernel(points[j], points) * w[j] * w

    w_sqrt = np.sqrt(w)
    aux = (points, jnp.array(w_sqrt))
    K_diag = w * kernel(points, points)
    return rpcholesky(K_col_fn, aux, K_diag, rank,
                      rng_gen=rng_gen)


def rpcholesky(A_col_fn, aux, A_diag, rank,
               rng_gen=None,
               pivots=None,
               eps=1e-8):
    '''
    Randomly pivoted Cholesky.

    Args:
        A_col_fn: A function that returns the i-th column of A.
        aux:
            Auxiliary data for A_col_fn. This is not baked into A_col_fn
            because it may be large and can change during adaptation, and
            we want JAX to trace aux.
        A_diag: The diagonal of A.
        rank: The Nystrom rank.
        pivots: Optional, the pivots to use.
        rng_gen: Random number generator.
        eps: The threshold for numerical stability.
    Returns:
        F: n by k matrix such that A is approximated by F F^T.
        pivots: Indices of the pivots, if not given in the input.
    '''
    S_given = pivots is not None
    if pivots is None:
        pivots = jnp.zeros(rank, dtype=int)
    if rng_gen is None:
        rng_gen = np.random.default_rng()
    rng_key = jax.random.PRNGKey(rng_gen.integers(0, 2**32))
    return rpcholesky_impl(A_col_fn, aux, A_diag, rank, pivots, rng_key,
                           S_given=S_given, eps=eps)


@partial(jax.jit, static_argnames=('rank', 'A_col_fn', 'S_given'))
def rpcholesky_impl(A_col_fn, aux, A_diag, rank, S, rng_key,
                    S_given=False,
                    eps=1e-8):
    n = A_diag.shape[0]
    F = jnp.zeros((n, rank))
    pivots = jnp.zeros(rank, dtype=int)
    d = A_diag

    def loop_body(i, args):
        F, d, pivots, rng_key = args
        if S_given:
            s = S[i]
        else:
            rng_key, cur_key = jax.random.split(rng_key)
            s = jax.random.choice(cur_key, n, p=d / d.sum())
        g = A_col_fn(s, aux) - (F * F[s]).sum(1)
        changed = g[s] > eps
        F = jnp.where(changed, F.at[:, i].add(g / (jnp.sqrt(g[s])+eps)), F)
        if not S_given:
            d = jnp.where(changed, jnp.maximum(0, d - F[:, i] ** 2), d)
            pivots = pivots.at[i].set(s)
        return F, d, pivots, rng_key

    F, d, pivots, rng_key = jax.lax.fori_loop(
        0, rank, loop_body, (F, d, pivots, rng_key))
    if S_given:
        return F
    else:
        return F, pivots


def nystrom_approx(points, kernel, pivots):
    '''
    Compute the Nystrom approximation of the kernel matrix given the pivots.

    Args:
        points: A SliceablePoints instance.
        kernel: A broadcastable kernel function.
        pivots: Indices of the pivots.
    '''

    K_SS = kernel(points[pivots, None], points[None, pivots]) # (k, k)
    K_SS = nearestPD(K_SS)
    F_SS = np.linalg.cholesky(K_SS) # (k, k)
    F_SS_inv = np.linalg.lstsq(
        F_SS, np.eye(F_SS.shape[0]), rcond=None)[0] # (k, k)

    # WARNING: The following line might cause memory issues when n is big.
    K_S = np.array(kernel(points[pivots, None], points[None, :])) # (k, n)

    # Compute F_SS^{-1} K_S.
    FT = F_SS_inv @ K_S  # (k, n)
    return FT.T
