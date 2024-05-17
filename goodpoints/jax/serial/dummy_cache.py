'''
A base class for caching of intermediate results.
It also stores a sequence of random seeds used so far for reproducibility.
'''

import numpy as np


class DummyCache:
    def __init__(self):
        # We use a sequence of random seeds so that we can control the
        # reproducibility for each stage of the pipeline, where each stage
        # will have its own seed.
        self.seed_seq = []

    def create_rng_gen(self):
        return np.random.default_rng(self.seed_seq)

    def append_seed(self, new_seed):
        self.seed_seq.append(new_seed)
        return self.create_rng_gen()

    def nonblocking_advance(self, stage, new_record):
        '''
        Non-blocking advance of the cache to a new stage with new records.

        Args:
            stage: A string representing the stage of the pipeline.
            new_record: A dictionary of new records to be added to the cache.
        '''
        pass

    def blocking_advance(self, stage, new_record, exec_fn):
        '''
        Blocking advance of the cache to a new stage with new records. This
        function returns the result of executing exec_fn. In a non-trivial
        implementation of the cache, this function either fetches the result
        from the cache or blocks and executes exec_fn.

        Args:
            stage: A string representing the stage of the pipeline.
            new_record: A dictionary to update the current record.
            exec_fn:
                A function to execute when the cache is not hit. It should
                return a dictionary to be saved in the cache.
        '''

        return exec_fn()

    def finalize(result_dict, self):
        '''
        Log the result dictionary to the cache for evaluation.

        Args:
            result_dict: A dictionary of results to be logged.
        '''
        pass
