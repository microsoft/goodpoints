'''
Generalized kernel swap with line search for improving MMD to P with
a mean-zero kernel. This is a generalization of the kernel swap algorithm
that also extends Stein Thinning to simplex and constant-preserving weights.
'''

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class MMDStats:
    '''
    A data structure for maintaining statistics about the MMD during
    kernel swap with line search.

    Attributes:
        w: (n,)
        grad: (n,), grad[j] is 2 sum_i w_i k(p_i, p_j).
        mmd: Scalar, sum_{ij} w_i w_j k(p_i, p_j).
    '''
    def __init__(self, w, grad, mmd):
        self.w = w
        self.grad = grad
        self.mmd = mmd

    def tree_flatten(self):
        return ((self.w, self.grad, self.mmd), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def zero(cls, n):
        return cls(jnp.zeros(n), jnp.zeros(n), 0)

    def scale(self, alpha):
        return MMDStats(self.w * alpha, self.grad * alpha,
                        self.mmd * alpha * alpha)

    def add(self, j, t, kernel, points):
        '''
        w' = w + t e_j.
        '''
        w = self.w.at[j].add(t)
        Kj = kernel(points[j], points)
        grad = self.grad + 2 * t * Kj
        mmd = (self.mmd + t * self.grad[j].sum() +
               t * t * Kj[j])
        return MMDStats(w, grad, mmd)


@partial(jax.jit, static_argnames=('kernel', 'weight_type', 'grow_until'))
def kernel_swap_ls_impl(kernel, points, w, supp, weight_type, grow_until):
    '''
    See kernel_swap_ls for the full description of the arguments.

    Returns:
        w: (n,), updated weight vector.
        supp_cnt:
            (n,), updated support count. supp_cnt[j] is the number of copies
            of j in the support. In the weighted case, supp_cnt[j] is 0 or 1.
    '''
    n = points.length
    n_supp = supp.shape[0]
    K_diag = kernel(points, points) # (n,)

    def loop_body(i, args):
        # Accumulate initial statistics.
        stats, = args
        j = supp[i]
        if weight_type == 'integer':
            stats = stats.add(j, 1. / n_supp, kernel, points)
        else:
            stats = stats.add(j, w[j], kernel, points)
        return (stats,)

    def loop_body2(i, args):
        # One iteration of swapping.
        stats, supp, supp_cnt = args

        if grow_until is None:
            j = supp[i]
            # Swap j out.
            if weight_type == 'integer':
                # For integer weights, swap out one at a time.
                # The weight is always the same.
                wj = 1. / n_supp
            else:
                wj = stats.w[j]
            stats = stats.add(j, -wj, kernel, points)
            stats = stats.scale(1 / (1 - wj))

            supp_cnt = supp_cnt.at[j].add(-1)

        # Compute line search weights for all points.
        A = stats.mmd
        B = stats.grad / 2
        C = K_diag
        if weight_type == 'integer':
            n_integer = (n_supp if grow_until is None
                          else i + n_supp + 1)
            alpha = jnp.full_like(B, 1. / n_integer)
        else:
            # Added stability fix here for simplex and cp versions of ST.
            alpha = (A - B) / (A - 2 * B + C +1.e-12) # (n,)
            if weight_type == 'simplex':
                # For the simplex case, clamping gives the argmin of the
                # linear search.
                alpha = jnp.where(alpha < 0, 0,
                                  jnp.where(alpha > 1, 1, alpha))
        new_mmd = (1-alpha)**2 * A + 2 * (1-alpha) * alpha * B + alpha**2 * C

        if weight_type != 'integer':
            # Prevent swapping in points that are already in the support.
            new_mmd = jnp.where(supp_cnt > 0, jnp.inf, new_mmd)

        # Swap in the best replacement with line search weight.
        k = jnp.argmin(new_mmd)
        stats = stats.scale(1 - alpha[k])
        stats = stats.add(k, alpha[k], kernel, points)

        supp_cnt = supp_cnt.at[k].add(1)
        return (stats, supp, supp_cnt)

    stats = MMDStats.zero(n)
    stats = jax.lax.fori_loop(0, n_supp, loop_body, (stats,))[0]

    supp_cnt = jnp.zeros(n, dtype=int)
    supp_cnt = supp_cnt.at[supp].add(1) # in JAX, at would update all indices
    num_itr = (n_supp if grow_until is None else grow_until - n_supp)
    stats, _, supp_cnt = jax.lax.fori_loop(0, num_itr, loop_body2,
                                           (stats, supp, supp_cnt))
    return stats.w, supp_cnt


def kernel_swap_ls(kernel, points, w, supp, weight_type,
                   grow_until=None):
    '''
    Generalized kernel swap with linear search for various weight types.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance containing input information.
        w: (n,), weight vector.
        supp:
            (m,), indices of the support of w. If weight_type == 'integer',
            then supp is the indices of the coreset (allowing repeats).
        weight_type: {'integer', 'simplex', 'cp'}, type of weights.
        grow_until:
            int, grow the support until reaching this size. If None, then
            only swap out points that are already in the support.
    Returns:
        w: (n,), updated weight vector.
        supp:
            (m,), updated support. If weight_type == 'integer', then supp
            contains the sorted indices of the coreset (allowing repeats).
    '''
    n = points.length
    w, supp_cnt = kernel_swap_ls_impl(kernel, points, w, supp,
                                      weight_type, grow_until)
    w = np.array(w)
    supp_cnt = np.array(supp_cnt)
    if weight_type == 'integer':
        # Unroll duplicates in the support.
        supp = []
        for i in range(n):
            for j in range(supp_cnt[i]):
                supp.append(i)
        supp = np.array(supp)
    else:
        supp = np.arange(n)[supp_cnt > 0]

    return w, supp
