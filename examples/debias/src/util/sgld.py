'''
Stochastic Gradient Langevin Dynamics by Welling and Teh (2011) and
Stochastic Gradient Fisher Scoring by Ahn et al. (2012).
'''

import numpy as np
import tqdm

def sgld(grad_log_p_prior,
         grad_log_p_model,
         theta_start, *,
         num_iter,
         num_data,
         base_step_size,
         base_time,
         decay_alpha,
         batch_size, rng_gen,
         eval_freq=5000,
         eval_fn=None):

    def score_fn(theta, inds):
        score = grad_log_p_model(theta, inds).sum(0) # (d,)
        score = score * num_data / batch_size
        score = score + grad_log_p_prior(theta)
        return score

    d = theta_start.shape[0]
    theta = theta_start
    points, scores = [], []
    inds = rng_gen.choice(num_data, batch_size, replace=False)
    cur_score = score_fn(theta, inds)
    pbar = tqdm.trange(num_iter)
    for i in pbar:
        if base_time == 'none':
            step = base_step_size
        else:
            step = base_step_size * ((base_time+i) ** (-decay_alpha))
        delta = (step / 2) * cur_score
        noise = rng_gen.normal(size=d)
        delta = delta + np.sqrt(step) * noise
        theta = theta + delta

        # Make sure cur_score is tracking the score of the current theta.
        inds = rng_gen.choice(num_data, batch_size, replace=False)
        cur_score = score_fn(theta, inds)
        if (eval_fn is not None) and ((i+1) % eval_freq == 0):
            eval_fn(i, theta)
        points.append(theta.copy())
        scores.append(cur_score.copy())

    return np.array(points), np.array(scores)


def sgfs(grad_log_p_prior,
         grad_log_p_model,
         theta_start, *,
         num_iter,
         num_data,
         step_size,
         batch_size, rng_gen,
         diag_inv=False,
         eval_freq=5000,
         eval_fn=None):

    d = theta_start.shape[0]
    gamma = (num_data + batch_size) / batch_size
    theta = theta_start
    I_t = np.zeros((d, d))
    points, scores = [], []
    pbar = tqdm.trange(num_iter)
    for i in pbar:
        inds = rng_gen.choice(num_data, batch_size, replace=False)
        grads = grad_log_p_model(theta, inds) # (b, d)
        grads_avg = grads.mean(0) # (d,)

        points.append(theta.copy())
        scores.append(num_data * grads_avg + grad_log_p_prior(theta))

        V = ((grads - grads_avg)[:, :, None] *
             (grads - grads_avg)[:, None, :]) # (b, d, d)
        V = V.sum(0) / (batch_size - 1)
        kappa_t = 1 / (i+1)
        I_t = (1 - kappa_t) * I_t + kappa_t * V
        B = gamma * I_t * num_data
        B += np.eye(d) # stability fix

        noise = rng_gen.normal(size=d)
        noise_factor = np.sqrt(2 / step_size)

        if diag_inv:
            noise = noise_factor * np.sqrt(np.diag(B)) * noise
        else:
            B_ch = np.linalg.cholesky(B)
            noise = (noise_factor * B_ch) @ noise

        grad_prior = grad_log_p_prior(theta)
        inv_cond_mat = gamma * num_data * I_t + 4 * B / step_size
        tmp_vec = grad_prior + num_data * grads_avg + noise
        if diag_inv:
            tmp_vec = tmp_vec / np.diag(inv_cond_mat)
        else:
            tmp_vec = np.linalg.inv(inv_cond_mat) @ tmp_vec
        theta = theta + 2 * tmp_vec
        if (eval_fn is not None) and ((i+1) % eval_freq == 0):
            msg = eval_fn(i, theta)
            pbar.set_description(msg)


    return np.array(points), np.array(scores)
