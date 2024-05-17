'''
Kernel Thinning algorithm and its subrountines by Dwivedi et al. (2021).
'''

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import logging

from goodpoints.jax.sliceable_points import SliceablePoints
from goodpoints.jax.mmd import compute_mmd, compute_K_mean
from goodpoints.jax.st import stein_thin
from goodpoints.jax.rounding import stratified_round, log2_ceil


def kt(kernel, points, w, rng_gen, *,
       inflate_size, out_size, delta=0.5):
    '''
    Kernel thinning algorithm from Li et al. (2024) targeting a mean-zero
    kernel and a weighted input.

    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance containing input information.
        w: Weights, n-dimensional vector.
        rng_gen: Random number generator state.
        inflate_size:
            Multiplicity with inflate_size/out_size being a power of 2.
        out_size: Output size.
        delta: Failure probability.
    Returns:
        A numpy array of indices of the coreset of length out_size.
    '''
    t = log2_ceil(inflate_size, out_size)
    seq = stratified_round(w, inflate_size, rng_gen)
    logging.info(f'KT splitting with {seq.size} points and thinning to '
                 f'{out_size} points...')
    coresets = kernel_split(kernel, points.subset(seq), t, delta,
                            rng_gen)
    coresets = [seq[coreset] for coreset in coresets]
    logging.info('KT-swap improvement...')
    coreset = kernel_swap(kernel, points, coresets,
                          rng_gen, mean_zero=True)
    return coreset


def kernel_split(kernel, points, t, delta, rng_gen):
    '''
    Thins inputs points x to floor(n/2^t) points using a kernel.
    This takes O(n^2) time.

    Args:
        kernel: A broadcastable JAX kernel function.
        points: A SliceablePoints instance containing input information.
        t: Thinning parameter.
        delta: Failure probability, delta_i = delta/n.
        rng_gen: Random number generator state.
    Returns:
        A list of coresets, each a numpy integer array of size floor(n/2^t).
    '''
    n = points.length
    assert(2**t <= n)

    cur_list = [np.arange(n)]
    for i in range(t):
        next_list = []
        for j, coreset in enumerate(cur_list):
            coreset0, coreset1 = kernel_halve(
                kernel,
                points.subset(coreset),
                delta,
                rng_gen)
            next_list.append(coreset[coreset0])
            next_list.append(coreset[coreset1])
        cur_list = next_list

    assert(len(cur_list) == 2**t)
    return cur_list


def kernel_halve(kernel, points, delta, rng_gen,
                 block_size=16):
    '''
    Exact halving using a kernel.

    Args:
        kernel: A broadcastable JAX kernel function.
        points: A SliceablePoints instance containing input information.
        delta: Failure probabilities, delta_i = delta/n.
        rng_gen: Random number generator state.
        block_size: Block size for the intermediate kernel matrix.
    Returns: (coreset0, coreset1), where
        coreset0: A numpy array containing indices of the first half.
        coreset1: A numpy array containing indices of the second half.
    '''

    n = points.length
    rand = rng_gen.random(size=n//2)
    rand = jnp.array(rand)

    n_remain = n % (2*block_size)
    n_pad = (2*block_size - n_remain) % (2*block_size)

    # Add padding to the points. These points will not affect the result
    # since they are added to the end of the sequence.
    points_padded = points.pad(n_pad)
    mask0 = kernel_halve_impl(kernel, points_padded, delta, rand,
                              block_size=block_size)
    if n_pad > 0:
        mask0 = np.array(mask0[:-n_pad])
    coreset0 = np.arange(n)[mask0]
    coreset1 = np.arange(n)[np.logical_not(mask0)]

    return coreset0, coreset1


@partial(jax.jit, static_argnames=('kernel', 'block_size'))
def kernel_halve_impl(kernel, points, delta, rand,
                      block_size):
    '''
    JAX-compiled version of kernel_halve.
    To save the number of calls to the kernel function while reducing the
    memory footprint, we access a block of entries of the kernel matrix with
    a given block size.
    '''
    n = points.length
    assert(n % (block_size * 2) == 0)
    if block_size == -1:
        block_size = n // 2
    K_diag = kernel(points, points)
    K_parity = kernel(points[::2], points[1::2])

    def inner_loop_body(j, args):
        k, mask0, sigma_sqr, K_coreset0, K_sum, K_block = args
        i = k * block_size + j
        b_sqr = K_diag[2*i] + K_diag[2*i+1] - 2 * K_parity[i]
        a, sigma_sqr = get_swap_params(sigma_sqr, b_sqr, delta)
        alpha = 0
        alpha += K_sum[2*i] - K_sum[2*i+1]
        alpha += -2 * (K_coreset0[2*i] - K_coreset0[2*i+1])
        threshold = jnp.minimum(1., 0.5 * jnp.maximum(0, 1 - alpha / a))
        swap = rand[i] < threshold
        l0 = jnp.where(swap, 2*i+1, 2*i)
        l1 = jnp.where(swap, 2*i, 2*i+1)
        mask0 = mask0.at[l0].set(True)

        Kl0 = K_block[l0 % (2*block_size)]
        Kl1 = K_block[l1 % (2*block_size)]
        K_coreset0 = K_coreset0 + Kl0
        K_sum = K_sum + Kl0 + Kl1
        return (k, mask0, sigma_sqr, K_coreset0, K_sum, K_block)

    def outer_loop_body(k, args):
        mask0, sigma_sqr, K_coreset0, K_sum = args
        i = k * 2*block_size
        points_block = points.dynamic_slice(i, 2*block_size)
        K_block = kernel(points_block[:, None], points[None, :])
        _, mask0, sigma_sqr, K_coreset0, K_sum, _ = jax.lax.fori_loop(
            0, block_size, inner_loop_body,
            (k, mask0, sigma_sqr, K_coreset0, K_sum, K_block))
        return (mask0, sigma_sqr, K_coreset0, K_sum)

    mask0 = jnp.zeros((n,), dtype=bool)
    mask0 = jax.lax.fori_loop(0, n // (block_size * 2),
                              outer_loop_body,
                              (mask0, 0,
                               jnp.zeros((n,)), jnp.zeros((n,))))[0]

    return mask0


def kernel_swap(kernel, points, coresets, rng_gen,
                num_repeat=1,
                random_swap_order=False,
                baseline=True,
                mean_zero=True):
    '''
    First choose the coreset that has the smallest MMD among all given
    corests, and then greedily swap points from the coreset to minimize MMD,
    allowing repeats.
    Args:
        kernel: A broadcastable kernel function.
        points: A SliceablePoints instance containing input information.
        coresets: A list of coresets.
        rng_gen: Random number generator state.
        num_repeat: Number of times to repeat the greedy swapping.
        random_swap_order: If True, randomly permute the order of swapping.
        baseline:
            If True, add in either Stein thinning baseline (if mean-zero), or
            i.i.d. baseline otherwise.
        mean_zero:
            If True, the kernel is assumed to have zero mean in P and the swap
            is to minimize the MMD to P.
    Returns:
        (m,) a numpy array of indices of the improved coreset.
    '''
    n = points.length

    if not mean_zero:
        # Precompute K_sum in n^2 time.
        K_mean = compute_K_mean(kernel, points, points, jnp.ones((n,)) / n)
    else:
        K_mean = jnp.zeros((n,))

    def select_best_coreset(coresets):
        # Select the best coreset.
        coreset = np.copy(coresets[0])
        if len(coresets) > 1:
            cur_mmd = np.inf
            for next_coreset in coresets:
                next_mmd = compute_mmd(kernel,
                                       points.subset(next_coreset),
                                       mode='mean-zero')
                if not mean_zero:
                    # Adjust the MMD to be against all points.
                    next_mmd += -2 * K_mean[improved_coreset].mean()
                if next_mmd < cur_mmd:
                    coreset = np.copy(next_coreset)
                    cur_mmd = next_mmd
        return coreset

    candidates = [select_best_coreset(coresets)]
    if baseline:
        if mean_zero:
            baseline_coreset = stein_thin(
                kernel, points, coresets[0].shape[0])[1]
        else:
            baseline_coreset = np.linspace(
                0, n-1, coresets[0].shape[0], dtype=int)
        candidates.append(baseline_coreset)

    K_diag = kernel(points, points)
    # Next use KT-swap to improve each candidate coreset.
    for i, coreset in enumerate(candidates):
        m = coreset.shape[0]
        for r in range(num_repeat):
            if random_swap_order:
                order = rng_gen.permutation(m)
                coreset = coreset[order]
            coreset = kernel_swap_impl(
                kernel, points, K_diag, coreset, K_mean)
            coreset = np.sort(np.array(coreset)) # retain order
        candidates[i] = coreset

    return select_best_coreset(candidates)


@partial(jax.jit, static_argnames=('kernel',))
def kernel_swap_impl(kernel, points, K_diag, coreset, K_mean):
    m = coreset.shape[0]
    def loop_body(i, args):
        E, = args
        j = coreset[i]
        E = E + kernel(points[j], points)
        return (E,)

    def loop_body2(i, args):
        E, coreset = args
        j = coreset[i]

        # First, remove j.
        E -= kernel(points[j], points)
        # Next, find the best replacement for j. Be careful with scaling.
        diff = (K_diag + 2 * E) / m - 2 * K_mean
        # diff[i] is the change in MMD after adding in i.
        k = jnp.argmin(diff)
        E += kernel(points[k], points)
        coreset = coreset.at[i].set(k)
        return (E, coreset)

    n = points.length
    E = jax.lax.fori_loop(0, coreset.shape[0], loop_body,
                          (jnp.zeros((n,)),))[0]
    # E[i] = sum_{j} k(x_i, x'_j) for x'_j in coreset.
    _, coreset = jax.lax.fori_loop(0, coreset.shape[0], loop_body2,
                                   (E, coreset))
    return coreset


def get_swap_params(sigma_sqr, b_sqr, delta):
    b = jnp.sqrt(b_sqr)
    sigma = jnp.sqrt(sigma_sqr)
    a = jnp.maximum(b * sigma * jnp.sqrt(2 * jnp.log(2 / delta)), b_sqr)
    a = jnp.maximum(a, 1.e-10) # stability fix
    sigma_sqr = sigma_sqr + b_sqr * jnp.maximum(
        0,
        1 + (b_sqr-2*a)*sigma_sqr/(a**2))
    return a, sigma_sqr
