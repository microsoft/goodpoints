'''
Negative Euclidean distance kernel k(x,, y) = -||x - y||_M where
||x-y||^2_M = (x-y)^T M^{-1} (x-y).
This is useful for energy distance computations.
'''

import jax
from functools import partial
import jax
import jax.numpy as jnp

from goodpoints.jax.sliceable_points import SliceablePoints
from goodpoints.jax.dot import mat_vec_dot


class NegEuclideanKernel:
    def __init__(self, M_inv):
        '''
        Args:
            M_inv: (d, d), the inverse of the M matrix.
        '''
        self.M_inv = M_inv

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, points_x, points_y):
        x = points_x.get('p')
        y = points_y.get('p')
        diff = x - y
        tmp = mat_vec_dot(self.M_inv, diff)
        return -jnp.sqrt((diff * tmp).sum(-1))

    def prepare_input(self, p):
        return SliceablePoints({'p': p})
