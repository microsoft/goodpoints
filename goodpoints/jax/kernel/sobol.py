'''
Sobolev space kernel on [0,1]^d that is mean-zero with respect to the
uniform measure.
'''

import jax
from functools import partial
import jax
import jax.numpy as jnp

from goodpoints.jax.sliceable_points import SliceablePoints

class SobolKernel:
    def __init__(self,
                 dim,
                 s):
        '''
        Args:
            dim: int, the dimension of the input space.
            s: int, 1 or 3, the Sobolev space index.
        '''
        self.dim = dim
        assert(s == 1 or s == 3)
        self.s = s

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, points_x, points_y):
        x = points_x.get('p')
        y = points_y.get('p')

        d = x - y
        d = jnp.where(d < 0, d + 1, d)
        d_sqd = d ** 2
        d_to_4 = d_sqd ** 2

        pi_sqd = jnp.pi ** 2
        pi_to_6 = jnp.pi ** 6

        if self.s == 1:
            ans = (1 + 2 * pi_sqd * (d * d - d + 1./6))
        else:
            ans = (1 + pi_to_6 * 4./45. * (
                (d_sqd*d_to_4) - 3 * (d*d_to_4) +
                5./2. * d_to_4 - d_sqd / 2 + 1. / 42
            ))


        ans = ans.prod(-1)

        return ans - 1 # mean 0

    def prepare_input(self, p):
        return SliceablePoints({'p': p})
