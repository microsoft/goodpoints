'''
Cholesky Thinning algorithm from Li et al. (2024) that produces a
constant-preserving coreset given weighted points.
'''

import numpy as np
import jax.numpy as jnp
import jax
import logging

from goodpoints.jax.rpc import weighted_rpcholesky
from goodpoints.jax.reopt import reopt_cp
from goodpoints.jax.kt_swap_ls import kernel_swap_ls


def chol_thin(kernel, points, w, rng_gen, *,
              out_size,
              c='auto'):
    '''
    Cholesky Thinning.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance with input information.
        w: Weights, n-dimensional vector.
        out_size: Output size.
        rng_gen: Random number generator.
        c:
            The constant for the shifted kernel. If 'auto', it is set to
            the average of the top out_size eigenvalues of the kernel matrix.

    '''
    n = points.length
    if c == 'auto':
        K_diag = kernel(points, points)[w > 0]
        K_diag = jnp.sort(K_diag)[::-1]

        # Set c to be the constant with the guarantee that
        # MMD_kp(w^1) <= 2 MMD_kc(w^c), where w^c is the best
        # unconstrained weight for the shifted kernel kp+c, and
        # recall MMD_kc(w^c) <= MMD_kc(w^c,w^*) + MMD_kp(w^*), and that
        # RPC gives a guarantee to bound MMD_kc(w^c,w^*).
        c = K_diag[:out_size].sum() / out_size

        logging.info(f'Auto-selected c={c}.')

    def kernel_regularized(x, y):
        return (c + kernel(x, y))

    _, supp = weighted_rpcholesky(kernel_regularized, points, w, out_size,
                               rng_gen)
    logging.info('Finished selecting support. Next computing the optimal '
                 'constant-preserving weight.')

    w_supp = reopt_cp(kernel, points.subset(supp))
    w = np.zeros(n)
    w[supp] = w_supp

    logging.info('KT-swap improvement...')
    w, supp = kernel_swap_ls(kernel, points, w, supp, 'cp')

    logging.info('Finally compute the optimal '
                 'constant-preserving weight again.')
    w_supp = reopt_cp(kernel, points.subset(supp))
    w = np.zeros(n)
    w[supp] = w_supp

    return w, supp
