'''
Preconditioned Stein kernel with a radially analytic base kernel.
'''

import numpy as np
import jax
from functools import partial
import jax
import jax.numpy as jnp

from goodpoints.jax.dot import mat_vec_dot
from goodpoints.jax.sliceable_points import SliceablePoints

class PrecondSteinKernel:
    def __init__(self,
                 scalar_kernel,
                 M,
                 med_sqr):
        '''
        k_base(x,y) = scalar_kernel((x-y)^T M^{-1} (x-y)/med_sqr)

        Args:
            scalar_kernel: analytic radial kernel function.
            M: (d, d), preconditioning PSD matrix.
            med_sqr: scalar, median squared distance.
        '''
        phi = lambda t: scalar_kernel(t / med_sqr)
        self.phi = jnp.vectorize(phi)
        self.dphi = jnp.vectorize(jax.grad(phi))
        self.ddphi = jnp.vectorize(jax.grad(jax.grad(phi)))
        self.dim = M.shape[0]
        self.M = M
        self.M_inv = np.linalg.inv(M)
        self.tr_M = jnp.trace(M)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, points_x, points_y):
        x, y = points_x.get('p'), points_y.get('p')
        s_x, s_y = points_x.get('s'), points_y.get('s')
        Ms_x, Ms_y = points_x.get('Ms'), points_y.get('Ms')
        LT_x, LT_y = points_x.get('LT_p'), points_y.get('LT_p')

        LT_xy_norm_sqr = ((LT_x - LT_y) ** 2).sum(-1)
        xy_norm_sqrs = ((x - y) ** 2).sum(-1)
        phi = self.phi(LT_xy_norm_sqr)
        dphi = self.dphi(LT_xy_norm_sqr)
        ddphi = self.ddphi(LT_xy_norm_sqr)

        res = phi * (Ms_x * s_y).sum(-1)
        res -= 2 * dphi * ((s_x - s_y) * (x - y)).sum(-1)
        res -= 2 * dphi * self.dim
        res -= 4 * ddphi * LT_xy_norm_sqr
        return res

    def prepare_input(self, points, scores):
        '''
        Create a SliceablePoints instance given input points positions and
        scores. The resulting instance will be used for kernel evaluations.
        '''
        L = np.linalg.cholesky(self.M_inv)
        Ms = mat_vec_dot(self.M, scores)
        LT_p = mat_vec_dot(L.T, points)
        return SliceablePoints({
            'p': points,
            's': scores,
            'Ms': Ms,
            'LT_p': LT_p,
        })
