'''
MMD computations using tiling and allowing for different modes.
'''

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def compute_mmd(kernel, points1, w1=None,
                points2=None, w2=None,
                batch_size=32,
                mode='mean-zero'):
    '''
    Wrapper for compute_mmd_impl to allow optional arguments.
    '''

    if w1 is None:
        w1 = jnp.ones([points1.length]) / points1.length
    if points2 is None:
        assert(w2 is None and mode == 'mean-zero')
        points2 = points1 # dummy
        w2 = w1 # dummy
    else:
        assert(mode != 'mean-zero')
        if w2 is None:
            w2 = np.ones([points2.length]) / points2.length

    return compute_mmd_impl(kernel, points1, points2, w1, w2,
                            batch_size=batch_size, mode=mode)


@partial(jax.jit, static_argnames=('kernel', 'batch_size', 'mode'))
def compute_mmd_impl(kernel, points1, points2, w1, w2,
                     batch_size=32,
                     mode='mean-zero'):
    '''
    Compute the squared MMD between two point sets allowing batching.

    Args:
        kernel: A broadcastable kernel function.
        points1, points2:
            SliceablePoints instances containing
            input information. It is assumed that points1 is smaller
            than points2.
        w1, w2: Weights for point sets.
        batch_size: Batch size for the computation.
        mode: ['mean-zero', 'trunc', 'full'], where, if we denote points1
            and points2 as x and x', respectively:
            'mean-zero' computes sum_{i,j} k(x_i, x_j) w_i w_j.
            'trunc' computes, in addition to mean-zero, the cross term
                -2sum_{i,j} k(x_i, x'_j) w_i w'_j.
            'full' computes, in addition to trunc,
                sum_{i,j} k(x'_i, x'_j) w'_i w'_j.
    '''
    n1 = points1.length
    pad1 = (batch_size - n1 % batch_size) % batch_size
    points1 = points1.pad(pad1)
    w1 = jnp.concatenate([w1, jnp.zeros([pad1])], 0)
    if mode != 'mean-zero':
        n2 = points2.length
        pad2 = (batch_size - n2 % batch_size) % batch_size
        points2 = points2.pad(pad2)
        w2 = jnp.concatenate([w2, jnp.zeros([pad2])], 0)

    res = (w1 * compute_K_mean(kernel, points1, points1, w1,
                               batch_size=batch_size)).sum()
    if mode != 'mean-zero':
        res += -2 * (w1 * compute_K_mean(kernel, points1, points2, w2,
                                         batch_size=batch_size)).sum()
        if mode == 'full':
            res += (w2 * compute_K_mean(kernel, points2, points2, w2,
                                        batch_size=batch_size)).sum()
    return res


@partial(jax.jit, static_argnames=('kernel', 'batch_size'))
def compute_K_mean(kernel, points1, points2, w2,
                   batch_size=32):
    '''
    Compute the kernel mean between two point sets allowing batching.
    The batching occurs along the first dimension of points2.

    Args:
        kernel: A broadcastable kernel function.
        points1, points2:
            SliceablePoints instances containing input information.
        w2: Weights for points2.

    Returns:
        A vector of length points1.length, where the i-th element is
        sum_{j} k(x_i, x'_j) w'_j.
    '''
    n1, n2 = points1.length, points2.length

    pad2 = (batch_size - n2 % batch_size) % batch_size
    points2 = points2.pad(pad2)
    w2 = jnp.concatenate([w2, jnp.zeros([pad2])], 0)

    def loop_body(k, args):
        i = k * batch_size
        points1, points2, w2, res = args
        points2_block = points2.dynamic_slice(i, batch_size)
        w2_block = jax.lax.dynamic_slice(w2, (i,), (batch_size,))
        K = kernel(points1[:, None], points2_block[None, :])
        K = K * w2_block[None]
        res += K.sum(1)
        return points1, points2, w2, res

    res = jax.lax.fori_loop(0, n2 // batch_size, loop_body,
                            (points1, points2, w2,
                             jnp.zeros([n1])))[-1]
    return res
