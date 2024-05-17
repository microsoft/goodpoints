'''
Utility functions for saving dictionaries to HDF5 files.
'''

import jax
import numpy as np
import h5py
from pathlib import Path

def save_dict_h5(save_dict, h5_path, create_dir=False):
    def recurse(remain_dict, parent_handle):
        for k, v in remain_dict.items():
            if isinstance(v, dict):
                child_handle = parent_handle.create_group(k)
                recurse(v, child_handle)
            else:
                if isinstance(v, jax.numpy.ndarray):
                    arr = np.array(v)
                elif isinstance(v, np.ndarray):
                    arr = v
                else:
                    # Must be scalar. Save as attributes.
                    parent_handle.attrs[k] = v
                    continue
                parent_handle.create_dataset(k, data=arr)
    if create_dir:
        Path(h5_path).parent.mkdir(parents=True, exist_ok=True)
    root_handle = h5py.File(h5_path, 'w')
    recurse(save_dict, root_handle)
