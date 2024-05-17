'''
Low-rank Debiasing algorithm from Li et al. (2024).
'''

import numpy as np
import logging
import jax
import jax.numpy as jnp

from goodpoints.jax.lr_matrix import LowRankKernelMatrix
from goodpoints.jax.amd import amd
from goodpoints.jax.rounding import stratified_round, seq_to_w_supp

def lr_debias(kernel, points, rng_gen, *,
              rank, num_iter,
              num_adapt=3,
              rounding=True,
              correct_diag=True,
              use_float64=False,
              nys_approx=False,
              step_size='inv_avg_diag',
              restart=True,
              log_freq=50):
    '''
    Low-rank debasing.

    If you encounter numerical issues, try enabling
        jax.config.update("jax_enable_x64", True)
    at start to use 64-bit floating point numbers.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance.
        rng_gen: Random number generator.
        rank: The rank of the low-rank approximation.
        num_iter: Number of iterations for AMD.
        num_adapt: Number of adaptive iterations.
        rounding:
            Whether to round the weights at the beginning of each adaptive
            iteration.
        correct_diag: Whether to correct the diagonal.
        use_float64: Whether to use float64 for storing low-rank matrices.
        nys_approx:
            Whether to compute Nystrom approximation explicitly when building
            the low-rank kernel matrix.
        step_size:
            Step size for AMD. If 'inv_avg_diag', use 1/(8 * avg(K_diag)).
        restart: Whether to restart AMD.
        log_freq: Logging frequency.
    '''
    n = points.length

    # Initialize with the uniform distribution.
    w = np.ones(n) / n
    supp = np.arange(n) # keep track of support

    @jax.jit
    def Kw_fn(w, K_lr):
        return K_lr.dot_single(w)

    @jax.jit
    def obj_fn(w, K_lr):
        return (w * K_lr.dot_single(w)).sum()

    specified_step_size = step_size
    for adapt_itr in range(num_adapt):
        if rounding and adapt_itr > 0:
            rounded_seq = stratified_round(w, n, rng_gen)
            w, supp = seq_to_w_supp(rounded_seq, n)
            logging.info(f'[Adapt #{adapt_itr}] Support size after '
                         f'rounding: {len(supp)}/{n}')

        logging.info(f'[Adapt #{adapt_itr}] Building low-rank kernel matrix...')
        K_diag = kernel(points[supp], points[supp])
        logging.info(f'[Adapt #{adapt_itr}] K_diag '
                     f'min: {K_diag.min()}, max: {K_diag.max()} '
                     f'mean: {K_diag.mean()}, median: {np.median(K_diag)}')
        lr_matrix = LowRankKernelMatrix.build(
            kernel,
            points.subset(supp),
            w=w[supp],
            rank=rank,
            rng_gen=rng_gen,
            correct_diag=correct_diag,
            use_float64=use_float64,
            nys_approx=nys_approx)

        K_hat_diag = lr_matrix.low_rank_diag()
        tr_ratio = K_hat_diag.sum() / K_diag.sum()
        logging.info('Complete building low-rank kernel. '
                     f'tr(K_hat)/tr(K) = {tr_ratio:.4f}')

        if specified_step_size == 'inv_avg_diag':
            step_size = 1. / (8 * (K_diag * w[supp]).sum().item())
        else:
            step_size = specified_step_size
        # Avoid the case where the step size is too small by choosing
        # the step size to be the upper bound in the analysis.
        # L smoothness constant = 2 * K_diag.max(), gamma = 1/4L by
        # Wang & Abernethy Alg 14.
        min_step_size = 1. / (8 * K_diag.max().item())

        # Increase step size if the initial step size is too small.
        step_size = max(min_step_size, step_size)

        # For the first adaptive iteration, use the minimum step size to make
        # analysis go through.
        if adapt_itr == 0:
            step_size = min_step_size
        logging.info(f'Using step size {step_size} for low-rank debiasing.')

        w_init = w[supp]
        w_init = (w_init.astype(np.float64) if use_float64
                  else w_init.astype(np.float32))

        w_supp, loss_traj = amd(
            n, Kw_fn, aux=lr_matrix,
            step_size=step_size,
            num_iter=num_iter,
            restart=restart,
            min_step_size=min_step_size,
            w_init=w_init,
            log_freq=log_freq)

        if obj_fn(w_supp, lr_matrix) > obj_fn(w_init, lr_matrix):
            logging.warning('AMD did not decrease the objective function. '
                            'Step size too large? '
                            'Using the initial point instead.')
            w_supp = w_init

        # Renormalize to avoid numerical issues.
        w = np.zeros_like(w)
        w[supp] = np.array(w_supp)
        w = np.maximum(0, w)
        w = w / w.sum()

    return w, supp, loss_traj
