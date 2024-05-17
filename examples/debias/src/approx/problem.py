'''
Approximate Bayesian inference.
'''
import logging
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from .logistic import LogisticRegression
from ..util.sgld import sgld, sgfs


def compute_full_scores(grad_log_p_prior_jax,
                        grad_log_p_model_jax,
                        num_data,
                        thetas):
    def loop_body(i, state):
        thetas, scores = state
        theta = thetas[i]
        full_s = grad_log_p_prior_jax(theta)
        full_s += grad_log_p_model_jax(
            theta, jnp.arange(num_data),
        ).sum(0)
        scores = scores.at[i].set(full_s)
        return thetas, scores

    def impl(thetas, scores):
        return jax.lax.fori_loop(
            0, len(thetas), loop_body,
            (thetas, scores)
        )[1]
    return impl(thetas, jnp.zeros_like(thetas))


def create_model(cfg, cache):
    cache.nonblocking_advance(
        'problem/model', cfg)
    return LogisticRegression(
        cfg['prior'], cfg['data_name'], cfg['data_folder'],
        use_probit=cfg['use_probit'],
        subsample=cfg['subsample'])


def generate_gold(model, cfg):
    # Since HMC is slow, we provide an option to load the
    # gold samples.
    save_file = Path(cfg['save_file'])
    if save_file.exists():
        logging.info(f'Loading gold samples from '
                     f'{save_file}...')
        points_gold = np.load(save_file)
    else:
        logging.info('Generating gold samples...')
        points_gold = model.sample_gold(cfg['num_warmup'],
                                        cfg['num_sample'],
                                        rng_gen=np.random.default_rng(42))
        np.save(save_file, points_gold)
    return points_gold


def run_mcmc(model, cfg, cache):
    rng_gen = cache.append_seed(cfg['seed'])
    def exec_fn():
        batch_size = cfg['batch_size']
        if batch_size == 'full':
            batch_size = model.num_data
        batch_size = min(batch_size, model.num_data)

        rng_gen = cache.create_rng_gen()
        if cfg['sampler'] == 'sgld':
            points, scores = sgld(
                model.grad_log_p_prior,
                model.grad_log_p_model,
                model.sample_prior(rng_gen),
                num_iter=cfg['num_step']+cfg['num_warmup'],
                num_data=model.num_data,
                batch_size=batch_size,
                base_step_size=cfg['step_size'],
                base_time=cfg['base_time'],
                decay_alpha=cfg['decay_alpha'],
                rng_gen=rng_gen,
                eval_fn=getattr(model, 'eval_fn', None),
            )
        else:
            assert(cfg['sampler'] == 'sgfs')
            points, scores = sgfs(
                model.grad_log_p_prior,
                model.grad_log_p_model,
                model.sample_prior(rng_gen),
                num_iter=cfg['num_step']+cfg['num_warmup'],
                num_data=model.num_data,
                batch_size=batch_size,
                step_size=cfg['step_size'],
                diag_inv=cfg['diag_inv'],
                rng_gen=rng_gen,
                eval_fn=getattr(model, 'eval_fn', None),
            )

        points = points[cfg['num_warmup']:]
        scores = scores[cfg['num_warmup']:]

        # Thinning.
        inds = np.linspace(0, len(points)-1, cfg['num_sample'], dtype=int)
        points = points[inds]
        scores = scores[inds]
        return {
            'points': points,
            'scores': scores,
        }

    result = cache.blocking_advance(
        'problem/mcmc', cfg, exec_fn
    )
    points, scores = result['points'][:], result['scores'][:]
    logging.info(f'MCMC chain after thinning has length {len(points)}.')
    return points, scores


def do_post(model, cfg, full_points, full_scores, cache):
    def exec_fn():
        assert(np.all(np.isfinite(full_scores)))

        num_sample = full_points.shape[0]
        num_point = cfg['num_point']
        if num_point == -1:
            num_point = num_sample

        # Standard thinning.
        inds = np.linspace(0, num_sample-1, num_point, dtype=int)
        points = full_points[inds]
        scores = full_scores[inds]

        if cfg['full_batch_score']:
            logging.info('Computing full-batch scores...')
            scores = np.array(compute_full_scores(
                model.grad_log_p_prior_jax,
                model.grad_log_p_model_jax,
                model.num_data,
                points))

        return {
            'points': points,
            'scores': scores,
        }
    result = cache.blocking_advance(
        'problem/post', cfg, exec_fn)
    return result['points'][:], result['scores'][:]


class ApproximateBayesian:
    def __init__(self, cfg, cache):
        model = create_model(cfg['model'], cache)

        points_gold = generate_gold(model, cfg['gold'])
        points, scores = run_mcmc(model, cfg['mcmc'], cache)
        points, scores = do_post(model, cfg['post'], points, scores, cache)

        self.model = model
        self.points_gold = points_gold
        self.points = points
        self.scores = scores

    def compute_full_scores(self, thetas):
        return compute_full_scores(
            self.model.grad_log_p_prior_jax,
            self.model.grad_log_p_model_jax,
            self.model.num_data,
            thetas,
        )

    def compute_hessian(self, theta):
        def grad_log_p(theta):
            tmp = self.model.grad_log_p_prior_jax(theta)
            tmp += self.model.grad_log_p_model_jax(
                theta, jnp.arange(self.model.num_data),
            ).sum(0)
            return tmp

        return jax.jacfwd(grad_log_p)(theta)
