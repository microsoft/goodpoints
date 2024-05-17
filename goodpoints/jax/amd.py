'''
Accelerated mirror descent on min w^T Kw subject to the simplex constraint.
'''

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial


def bregman_neg_ent(x, g):
    '''
    Bregman projection for solving
        argmin_{x' \in \simplex} <x', g> + B_\psi(x';x)
    where \psi is the negative entropy function.
    '''
    g = g - g.min() # for stability
    x = x * jnp.exp(-g)
    x = x / x.sum()
    return x


@partial(jax.jit, static_argnames=('Kw_fn', 'restart'))
def amd_step(w, opt_state, aux, *,
             Kw_fn,
             restart,
             min_step_size,
             step_size_decay=0.95):
    '''
    One step of Nestrov's 1-memory accelerated mirror descent.

    Args:
        w: Current weights in the simplex.
        opt_state: Auxiliary states.
    Returns:
        A tuple (w, opt_state) where w is the updated weights and opt_state
        is the updated auxiliary states.
    '''
    t, v, Kv, Kw = opt_state['t'], opt_state['v'],\
        opt_state['Kv'], opt_state['Kw']
    step_size = opt_state['params']['step_size']
    beta_t = 2 / (t + 1)
    z = (1 - beta_t) * w + beta_t * v
    # We keep track of Kv, Kw to avoid computing gradients twice when
    # restart is employed (this uses the fact that obj_fn is w^T Kw).
    grad_z = 2 * ((1 - beta_t) * Kw + beta_t * Kv)
    g = t * step_size * grad_z
    v_next = bregman_neg_ent(v, g)
    Kv_next = Kw_fn(v_next, aux)
    w_next = (1 - beta_t) * w + beta_t * v_next
    Kw_next = (1 - beta_t) * Kw + beta_t * Kv_next
    last_obj = opt_state['last_obj']

    if restart:
        new_obj = (w_next * Kw_next).sum()
        improved = new_obj < last_obj
        improved = jnp.where(
            step_size < min_step_size,
            True, # Force improvement if step size is too small.
            improved)

        t, w_next, v_next, last_obj, Kw_next, Kv_next = jax.lax.cond(
            improved,
            lambda: (t, w_next, v_next, new_obj, Kw_next, Kv_next),
            lambda: (0, w, w, last_obj, Kw, Kw))

        step_size = jnp.where(
            improved,
            step_size,
            step_size * step_size_decay)

    opt_state = {
        'params': {
            'step_size': step_size,
        },
        't': t + 1,
        'v': v_next,
        'last_obj': last_obj,
        'Kv': Kv_next,
        'Kw': Kw_next,
    }
    return w_next, opt_state


def amd(dim, Kw_fn, *, aux,
        step_size, num_iter,
        restart,
        min_step_size,
        w_init=None,
        log_freq=50):
    '''
    Nestrov's 1-memory accelerated mirror descent on min w^T Kw subject to
    the simplex constraint. This corresponds to Algorithm 14 of
    Wang & Abernethy (2023).

    Args:
        dim: Dimension of the problem.
        Kw_fn: Function to compute the matrix-vector product Kw.
        aux: Auxiliary data for Kw_fn.
        step_size: Step size.
        num_iter: Number of iterations.
        Kw_fn: Function to compute the matrix-vector product Kw.
        restart:
            Whether to restart the algorithm when the objective increases.
        min_step_size:
            Minimum step size that guarantees progress based on the analysis.
        w_init: Optional initial weights.
    Returns:
        A tuple (w, loss_traj) where w is the final weights and loss_traj is
        the trajectory of the loss.

    '''
    if w_init is None:
        w = jnp.ones(dim) / dim
    else:
        w = w_init

    Kw = Kw_fn(w, aux)
    opt_state = {
        'params': {
            # Typecast in case we use float32
            'step_size': jnp.array(step_size,
                                   dtype=w.dtype),
        },
        't': 1, # starting at 1
        'v': w,
        'last_obj': (w * Kw).sum(),
        'Kv': Kw,
        'Kw': Kw,
    }

    loss_traj = []
    grad_norm_traj = []
    loss = 0

    pbar = tqdm(range(num_iter))

    for i in pbar:
        w, opt_state = amd_step(
            w, opt_state, aux,
            Kw_fn=Kw_fn,
            restart=restart,
            min_step_size=min_step_size,
        )

        if i % log_freq == 0:
            loss = (w * Kw_fn(w, aux)).sum().item()
            loss_traj.append((i, loss))
            msg = f'Loss: {loss:.6f}'
            msg += f', Step size: {opt_state["params"]["step_size"]:.6g}'
            pbar.set_description(msg)

    return w, loss_traj
