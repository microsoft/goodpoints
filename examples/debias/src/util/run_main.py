'''
Template for the main function for each experiment.
'''

import logging
import numpy as np
import jax
from pathlib import Path
import time
from omegaconf import OmegaConf
import wandb

from goodpoints.jax.serial.cache_factory import make_cache
from goodpoints.jax.dtc import dtc
from goodpoints.jax.mmd import compute_mmd
from goodpoints.jax.kernel.neg_euclidean import NegEuclideanKernel

from ..util.logging import \
    log_thinned_samples, log_weighted_samples
from ..util.parse import parse_n_expr, parse_n_m_dict
from ..util.misc import pop_cfg_name, pop_cfg_name_and_seed


def main_template(cfg,
                  make_problem_fn,
                  make_kernel_fn,
                  make_gold_params_fn=None,
                  make_full_kernel_fn=None):
    jax.config.update('jax_platforms', cfg.jax.platforms)
    jax.config.update('jax_enable_x64', cfg.jax.enable_x64)

    # root_dir should point to the parent of src/
    root_dir = Path(__file__).resolve().parent.parent.parent
    cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_run = wandb.init(config=cfg)

    record = {
    }
    non_key_record = {
        '_wandb': {
            'entity': wandb_run.entity,
            'project': wandb_run.project,
            'id': wandb_run.id
        }
    }
    cache_cfg = cfg['cache']
    cache_name, cache_cfg = pop_cfg_name(cache_cfg)
    cache = make_cache(cache_name, cache_cfg, record, non_key_record,
                       cache_dir=root_dir / 'cache')

    problem = make_problem_fn(cfg['problem'], cache)
    kernel = make_kernel_fn(problem, cache)
    points = kernel.prepare_input(problem.points, problem.scores)

    # Parsing expressions in the config.
    debias_alg, debias_seed, debias_cfg = pop_cfg_name_and_seed(
        cfg['debias'])
    compress_alg, compress_seed, compress_cfg = pop_cfg_name_and_seed(
        cfg['compress'])
    n = points.length
    m = parse_n_expr(cfg['out_size'], n)
    debias_cfg = parse_n_m_dict(debias_cfg, n, m)
    compress_cfg = parse_n_m_dict(compress_cfg, n, m)

    # Run the main algorithm.
    start_time = time.time()
    w, supp = dtc(kernel, points, cache,
        debias_alg=debias_alg, debias_cfg=debias_cfg,
        debias_seed=debias_seed,
        compress_alg=compress_alg, compress_cfg=compress_cfg,
        compress_seed=compress_seed)

    logging.info(f'Debiased compression takes {time.time() - start_time}s '
                 f'to finish.')

    start_time = time.time()
    result_dict = {}
    def log_fn(key, value):
        wandb.log({key: value}, step=0)
        result_dict[key] = value

    if make_gold_params_fn is not None:
        gold_params = make_gold_params_fn(problem)
    else:
        gold_params = None
    if make_full_kernel_fn is not None:
        full_kernel = make_full_kernel_fn(problem)
    else:
        full_kernel = None

    evaluate(kernel, points, w, supp,
             full_kernel=full_kernel,
             gold_params=gold_params,
             points_gold=problem.points_gold,
             log_fn=log_fn)
    logging.info(f'Evaluation takes {time.time() - start_time}s to finish.')
    cache.finalize(result_dict)


def evaluate(kernel, points, w, supp, *,
             full_kernel, gold_params, points_gold, log_fn):
    supp = np.unique(supp)

    logging.info('Original input size: {}; unique support size: {}'.format(
        points.length,
        len(supp)
    ))
    points, w = points.subset(supp), w[supp]
    coords = points.get('p')
    # Plotting final samples.
    logging.info('Logging final samples cmap...')
    log_weighted_samples(step=0, samples=coords,
                         weights=w,
                         wandb_key='final_samples_cmap',
                         style='cmap')
    logging.info('Logging final samples alpha...')
    log_weighted_samples(step=0, samples=coords,
                         weights=w,
                         wandb_key='final_samples_alpha',
                         style='alpha')
    logging.info('Logging gold samples...')
    log_weighted_samples(step=0, samples=points_gold,
                         wandb_key='gold_samples',
                         style='alpha')

    # Compute final MMD.
    logging.info('Computing final MMD...')
    final_mmd = compute_mmd(kernel, points1=points, w1=w)
    final_mmd = np.sqrt(final_mmd)
    log_fn('final_mmd', final_mmd)

    if full_kernel is not None:
        # Compute MMD (using full params).
        logging.info('Computing full MMD...')
        full_points = full_kernel.prepare_input(
            points.get('p'), points.get('s'))
        full_mmd = compute_mmd(full_kernel, points1=full_points, w1=w)
        full_mmd = np.sqrt(full_mmd)
        log_fn('full_mmd', full_mmd)

    if gold_params is not None:
        # Compute ED (without the E||X-X'|| term).
        logging.info('Computing gold ED...')
        M_inv = gold_params['M_inv']
        ed_kernel = NegEuclideanKernel(M_inv)
        ed_points = ed_kernel.prepare_input(coords)
        ed_gold_points = ed_kernel.prepare_input(points_gold)
        final_ed = compute_mmd(ed_kernel,
                               points1=ed_points, w1=w,
                               points2=ed_gold_points,
                               mode='trunc')
        final_ed = np.sqrt(final_ed)
        log_fn('final_ed', final_ed)

        # Compute MSE.
        logging.info('Computing MSE...')
        gold_mean = np.mean(points_gold, axis=0)
        points_mean = (w[:, None] * coords).sum(0)
        tmp = ((gold_mean - points_mean) *
               (M_inv @ (gold_mean - points_mean))).sum()
        mse = (tmp / coords.shape[1]).item()
        log_fn('mse', mse)
