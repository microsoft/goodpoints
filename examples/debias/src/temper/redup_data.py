import h5py
import numpy as np
import pandas as pd


point_handle = h5py.File('THETA_seed_1_temp_8.h5', 'r')
point_dedup_handle = h5py.File('THETA_unique_seed_1_temp_8.h5', 'r')
score_dedup_handle = h5py.File('GRAD_unique_seed_1_temp_8.h5', 'r')

point_dedup_csv = pd.read_csv('THETA_unique_seed_1_temp_8.csv',
                              header=None).to_numpy().T

points = point_handle['point'][:]
points_dedup = point_dedup_handle['point'][:]
scores_dedup = score_dedup_handle['score'][:]

n = points.shape[0]
warmup = n // 4
_, unique_inv = np.unique(points[warmup:], axis=0, return_inverse=True)
points_redup = points_dedup[unique_inv]
scores_redup = scores_dedup[unique_inv]

with h5py.File('points_scores_nowarmup.h5', 'w') as f:
    f.create_dataset('points', data=points_redup)
    f.create_dataset('scores', data=scores_redup)
