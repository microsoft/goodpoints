'''
Low-rank approximation of kernel matrices to speed up matrix-vector products.
'''

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from functools import partial

from goodpoints.jax.rpc import \
    rpcholesky, weighted_rpcholesky, nystrom_approx
from goodpoints.jax.dot import mat_vec_dot


@register_pytree_node_class
class LowRankKernelMatrix:
    def __init__(self, H, D):
        '''
        Form approximation matrix K' = H^T H + D that supports fast
        matrix-vector product.

        Args:
            H: (r, n) the low-rank factor.
            D: (n,) diagonal residual.
        '''
        self.H = H
        self.D = D

    def tree_flatten(self):
        return ((self.H, self.D), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def build(cls, kernel, points, rank, rng_gen,
              w=None,
              correct_diag=True,
              use_float64=False,
              nys_approx=False):
        '''
        Args:
            kernel: A broadcastable kernel function.
            points: A SliceablePoints instance.
            rank: Rank of the approximation.
            rng_gen: Random number generator state.
            w: (n,), Nystrom is applied to sqrt(diag(w)) K sqrt(diag(w)).
            correct_diag: Whether to correct the diagonal.
            use_float64: Whether to use float64 for storing low-rank matrices.
            nys_approx:
                Whether to explicitly construct Nystrom approximation after
                pivoting with numerical stablity fixes. If False, use
                RPCholesky (less numerically stable).
        '''
        n = points.length
        if w is None:
            w = np.ones(n)
        _, pivots = weighted_rpcholesky(kernel, points, w, rank, rng_gen)

        K_diag = kernel(points, points)
        def K_col_unweighted_fn(j, points):
            return kernel(points[j], points)
        if nys_approx:
            H = nystrom_approx(points, kernel, pivots)
        else:
            H = rpcholesky(K_col_unweighted_fn, points, K_diag, rank,
                           pivots=pivots)
        H = H.T # (r, n), low-rank factor

        H_diag = (H ** 2).sum(0)
        D = jnp.zeros(n)
        if correct_diag:
            D = jnp.maximum(0., K_diag - H_diag)

        # WARNING: we store matrix H in jax_dtype which could be float32
        # because otherwise JAX seems to have a bug or memory issue with storing
        # too large a matrix in float64 on GPU.
        dtype = jnp.float64 if use_float64 else jnp.float32
        H = jnp.array(H, dtype=dtype)
        D = jnp.array(D, dtype=dtype)
        return cls(H, D)

    @partial(jax.jit, static_argnums=0)
    def dot_single(self, w):
        '''
        Return K' w in O(nr) time.

        Args:
            w : (n,), a vector.
        '''
        res = jnp.dot(self.H, w) # (r,)
        res = jnp.dot(self.H.T, res) # (n,)
        res += self.D * w
        return res

    def low_rank_diag(self):
        return (self.H ** 2).sum(0)
