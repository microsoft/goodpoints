'''
This file contains the class for the on-disk problem where the points and
scores are stored on disk. The format is assumed to be the data format
provided by Riabiz et al. (2021).
'''
import logging
import pandas as pd
import numpy as np
import h5py
import yaml
from pathlib import Path


def read_csv_batched(path):
    return pd.read_csv(path, header=None).to_numpy()


def load_burnin(folder, level='none'):
    meta_file = folder / 'meta.yaml'
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
        if level == 'none':
            burnin = 0
        elif level == 'low':
            burnin = meta['burnin_low']
        else:
            assert(level == 'high')
            burnin = meta['burnin_high']
    else:
        burnin = 0
    return burnin


def filter_inf(points, scores):
    # Filter out NaNs in scores.
    mask = np.isfinite(scores).all(-1)
    points = points[mask]
    scores = scores[mask]
    bad_count = np.sum(~mask)
    if bad_count > 0:
        logging.info(f'Filtered out {bad_count} bad samples.')
    return points, scores


def load_from_folder(folder, cfg):
    logging.info('Reading points and scores data from disk...')
    if cfg['fmt'] == 'csv':
        points_all = read_csv_batched(folder / cfg['point_file'])
        scores_all = read_csv_batched(folder / cfg['score_file'])
        if points_all.shape[0] < points_all.shape[1]:
            # We always assume there are more samples than dimension.
            points_all = points_all.T
            scores_all = scores_all.T
    else:
        assert(cfg['fmt'] == 'h5')
        with h5py.File(folder / cfg['point_score_file'], 'r') as f:
            points_all = f['points'][:]
            scores_all = f['scores'][:]

    logging.info('Loaded points and scores data of shape '
                 f'{points_all.shape} from disk.')
    return points_all, scores_all


class OnDisk:
    def __init__(self, cfg, cache):
        cache.append_seed(cfg['seed'])
        cache.nonblocking_advance('problem', cfg)

        folder = Path(cfg['folder'])
        # Load data.
        points_full, scores_full = load_from_folder(folder, cfg)
        # Filter out NaNs in scores.
        points_full, scores_full = filter_inf(points_full, scores_full)
        # Remove burnin to form full points.
        burnin = load_burnin(folder, cfg['burnin'])
        self.points_full = points_full
        self.scores_full = scores_full

        gold_folder = folder / Path(cfg['gold_folder'])
        if gold_folder.exists():
            logging.info(f'Loading gold data from {gold_folder}...')
            points_gold, scores_gold = load_from_folder(
                gold_folder, cfg)
            gold_burnin = load_burnin(gold_folder, cfg['burnin'])
        else:
            logging.info(f'Gold data not found at {gold_folder}. '
                         f'Using full data as gold.')
            points_gold = points_full
            scores_gold = scores_full
            gold_burnin = burnin

        # Gold always has burnin removed.
        points_gold = points_gold[gold_burnin:]
        scores_gold = scores_gold[gold_burnin:]
        self.points_gold = points_gold
        self.scores_gold = scores_gold

        if cfg['remove_burnin']:
            points_ref = points_full[burnin:]
            scores_ref = scores_full[burnin:]
        else:
            points_ref = points_full
            scores_ref = scores_full
        num_data = points_ref.shape[0]
        num_sample = cfg['num_sample']
        if num_sample == -1:
            num_sample = num_data
        # Standard thinning (keep last point).
        indices = np.linspace(num_data-1, 0, num_sample,
                              endpoint=False, dtype=int)[::-1]

        points = points_ref[indices]
        scores = scores_ref[indices]

        self.points = points
        self.scores = scores
