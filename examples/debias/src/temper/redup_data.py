import h5py
import numpy as np
import pandas as pd
import os
import argparse

def get_h5_handle(file_basename, dataset_name):
    h5_filename = file_basename + '.h5'
    if not os.path.exists(h5_filename):
        csv_filename = file_basename + '.csv'
        # Load CSV file data
        print(f'Loading {csv_filename}')
        data = pd.read_csv(file_basename + '.csv',
                           header=None).to_numpy().T
        # Write to HDF5 file
        print(f'Writing {h5_filename}')
        with h5py.File(h5_filename, 'w') as hdf:
            # Write the data to the HDF5 file
            hdf.create_dataset(dataset_name, data=data)
    # Open HDF5 file for reading and return handle
    print(f'Loading {h5_filename}')
    return h5py.File(h5_filename, 'r')

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Creates a version of the deduplicated Cardiac points and scores arrays with repeats.')
    parser.add_argument(
        '-t', '--temp', default=8, type=int,
        help='The temperature of the tempered chain (1 for no tempering).')
    args = parser.parse_args()
    temp = args.temp

    # Load data
    point_handle = get_h5_handle(f'THETA_seed_1_temp_{temp}', 'point')
    point_dedup_handle = get_h5_handle(f'THETA_unique_seed_1_temp_{temp}', 'point')
    score_dedup_handle = get_h5_handle(f'GRAD_unique_seed_1_temp_{temp}', 'score')

    points = point_handle['point'][:]
    points_dedup = point_dedup_handle['point'][:]
    scores_dedup = score_dedup_handle['score'][:]

    # Add repeat points
    n = points.shape[0]
    warmup = n // 4
    _, unique_inv = np.unique(points[warmup:], axis=0, return_inverse=True)
    points_redup = points_dedup[unique_inv]
    scores_redup = scores_dedup[unique_inv]

    # Save points and scores together
    with h5py.File('points_scores_nowarmup.h5', 'w') as f:
        f.create_dataset('points', data=points_redup)
        f.create_dataset('scores', data=scores_redup)

if __name__ == '__main__':
    main()
