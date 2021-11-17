"""Compress, Compress++, and Symmetrize.

Implementations of the Compress, Compress++, and Symmetrize metaprocedures of
  Abhishek Shetty, Raaz Dwivedi, and Lester Mackey.
  Distribution Compression in Near-linear Time.
  https://arxiv.org/pdf/2111.07941.pdf
"""

import numpy as np
import numpy.random as npr

def compress(X, halve, g = 0, indices = None):
    """Returns Compress coreset of size 2^g sqrt(n) or, if indices is not None, 
    of size 2^g sqrt(len(indices)) as row indices into X

    Args:
        X: Input sequence of sample points with shape (n, d)
        halve: Algorithm that takes an input a set of points and returns indices a set of indices to a subset of points with cardinality half of the input set 
        g: Oversampling parameter, a nonnegative integer
        indices: If None, compresses X and returns coreset of size 2^g sqrt(n); 
          otherwise, compresses X[indices] and returns coreset of size 
          2^g sqrt(len(indices))
    """
    # Check if indices is None in which case it sets it to range(size(X))
    if indices is None:
        indices = np.arange(X.shape[0], dtype=int)
    # If the number of input points matches the target coreset size, we're done
    if len(indices) == 4**g:
        return indices
    else: 
        # Partition the set input indices into four disjoint sets
        partition_set = np.array_split(indices,4)
        # Initialize an array to hold outputs of recursive calls
        compress_outputs = []
        for x in partition_set:
            # Recursively call compress on the four subsets and
            # add its output to the list
            compress_output = compress(X, halve, g, indices = x)
            compress_outputs.append(compress_output)
        # Merge outputs of the recursive calls to compress
        combined_compress_output = np.concatenate(compress_outputs)
        # Run halving on the combined output
        indices_into_combined = halve(X[combined_compress_output])
        # Return indices into the original input set X
        return combined_compress_output[indices_into_combined]


def compresspp(X, halve, thin, g):
    """Returns Compress++(g) coreset of size sqrt(n) as row indices into X

    Args: 
        X: Input sequence of sample points with shape (n, d)
        halve: Function that takes in an (n', d) numpy array Y and returns 
          floor(n'/2) distinct row indices into Y, identifying a halved coreset
        thin: Function that takes in an (n', d) numpy array Y and returns
          2^g sqrt(n') row indices into Y, identifying a thinned coreset
        g: Oversampling factor
    """
    # Use compress to create a coreset of size 2^g sqrt(n)
    intermediate_coreset = compress(X, halve, g)
    # Use thin to reduce size from 2^g sqrt(n) to sqrt(n)
    return intermediate_coreset[ thin(X[intermediate_coreset]) ]

def symmetrize(halve, seed = None):
    """Converts a halving algorithm halve into a symmetrized halving algorithm.

    Args:
        halve: Function that takes in an (n', d) numpy array Y and returns 
          floor(n'/2) distinct row indices into Y, identifying a halved coreset
    
    Returns:
        A function that takes in an (n', d) numpy array Y and returns 
        either the row indices outputted by halve or the complementary
        row indices into Y, each with probability 1/2
    """
    def symmetrized_halve(X):
        """Symmetrized version of halve.
        
        Args: 
            X: Input sequence of sample points with shape (n, d)

        Returns:
            Either the row indices outputted by halve or the complementary
            row indices into X, each with probability 1/2
        """

        # define the output of halve
        halve_output = halve(X)
        # initialize rng
        rng = npr.default_rng(seed)
        # create a vector of boolean values
        n = len(X)
        mask = np.zeros(n, dtype=bool)
        # set mask values of halve output to one
        mask[halve_output] = True
        # check that size of output is half that of input
        assert(sum(mask)==int(n/2))
        if rng.choice([-1, 1]) == 1: # with probability half
            # take either mask or negation of mask
            mask = ~mask
        # return indices corresponding to mask 
        return np.where(mask)[0] 
    return symmetrized_halve
