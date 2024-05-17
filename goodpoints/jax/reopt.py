'''
Reoptimizatio with respect to a fixed support for simplex and
constant-preserving weigts.
'''

import numpy as np
import jax
import jax.numpy as jnp
from qpsolvers import solve_qp

def reopt_simplex(kernel, points, solver='proxqp',
                  initval=None):
    '''
    Compute the optimal simplex weight vector w using QP.
    Note that depending on the solver, you might need to install
    additional packages. For example, to use 'proxqp', you need to run
        pip install qpsolvers[proxqp]

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance of length n.
        solver: A string specifying the solver to use.
        initval: Initial value for the solver.
    Returns:
        w: (n,), reoptimized weight vector.
    '''
    n = points.length
    K = kernel(points[:, None], points[None, :])
    K_np = np.array(K)
    A = np.ones((n,))
    b = np.array([1.])
    lb = np.zeros((n,))
    q = np.zeros((n,))
    w = solve_qp(K_np, q, A=A, b=b, lb=lb, solver=solver,
                 initvals=initval)
    w = np.maximum(w, 0.) # Numerical errors can cause negative values.
    w /= w.sum()

    return w


def reopt_cp(kernel, points):
    '''
    Compute the optimal constant-preserving weight vector w using least square.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance of length n.
    Returns:
        w: (n,), reoptimized weight vector.
    '''
    n = points.length
    K = kernel(points[:, None], points[None, :])
    w = np.linalg.lstsq(K, np.ones(n), rcond=1e-12)[0]
    w = w / w.sum()
    return w
