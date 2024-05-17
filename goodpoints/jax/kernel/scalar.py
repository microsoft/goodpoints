'''
1D scalar kernels.
'''

import jax.numpy as jnp

def imq(t):
    return jnp.power(1 + t, -0.5)
