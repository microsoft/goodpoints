"""Compress, Compress++, and Symmetrize.

Implementations of the Compress, Compress++, and Symmetrize metaprocedures of
  Abhishek Shetty, Raaz Dwivedi, and Lester Mackey.
  Distribution Compression in Near-linear Time.
  https://arxiv.org/pdf/2111.07941.pdf
"""

import numpy as np
import numpy.random as npr
from goodpoints import compressc

def compress_gsn_kt(X, g = 0, num_bins=1, lam_sqd=np.ones(1), delta=0.5, seed=12345):
    """Returns coreset of size min(n, 2^{g+1} sqrt(n * num_bins)) as row indices into X.  
    Coreset is obtained by dividing the rows of X into num_bins consecutive bins,
    running Compress(g) on each bin with a Gaussian kernel and
    Halve = symmetrized kernel thinning, and concatenating the resulting per-bin 
    coresets.

	Note: Assumes n / num_bins is a power of 4.

    Args:
      X: Input sequence of sample points with shape (n, d)
      g: Oversampling parameter, a nonnegative integer
      num_bins: Number of bins, a positive integer <= n
      lam_sqd: Array of squared Gaussian kernel bandwidths to compute
        the sum-of-Gaussians kernel k(x,y) = sum_j exp(-||x-y||_2^2/lam_sqd[j])
      delta: Failure probability parameter for kernel thinning
      seed: Integer seed for initializing the random number generator used
        by Halve
    """
    # Allocate array for storing coreset indices
    n = X.shape[0]
    coreset_size = min(n, int(np.sqrt(n * num_bins)*(2**g)))    
    #print(f'n={n},num_bins={num_bins},g={g},coreset_size={coreset_size}')
    output_indices = np.empty(coreset_size, dtype=np.int64)
    
    # Run Compress on each bin
    if np.isscalar(lam_sqd):
        #print(f'lam_sqd scalar')
        lam_sqd = np.array([lam_sqd])
    compressc.compress(X, g, num_bins, lam_sqd, delta, seed, output_indices)
    return output_indices
     
def compress(X, halve, g = 0, indices = None):
    """Returns Compress coreset of size 2^g sqrt(n') as row indices into X.
    Here, n' is the largest power of 4 less than or equal to n (if indices 
    is None) or to len(indices) (if indices is not None).

    Note: When the number of indices to be compressed is not a power of 4, 
        the code reduces the index count to n' using standard thinning 
        (i.e., by selecting every t-th index) prior to running Compress.

    Args:
        X: Input sequence of sample points with shape (n, d)
        halve: Algorithm that takes an input a set of points and returns 
          indices a set of indices to a subset of points with cardinality 
          half of the input set 
        g: Oversampling parameter, a nonnegative integer
        indices: If None, compresses X and returns coreset of size 2^g sqrt(n); 
          otherwise, compresses X[indices] and returns coreset of size 
          2^g sqrt(len(indices))
    """
    # Check if indices is None in which case it sets it to range(size(X))
    if indices is None:
        indices = np.arange(X.shape[0], dtype=int)

    # If the number of indices is not a power of 4, thin down to the nearest power of 4
    # using standard thinning (i.e., by retaining every t-th index)
    n = len(indices)
    nearest_pow_four = largest_power_of_four(n)
    if nearest_pow_four != n:
        indices = indices[np.linspace(0, n-1, nearest_pow_four, dtype=int)]

    # Helper function for recursive calls to Compress
    four_to_g = 4**g
    def _compress(indices):
        # If the number of input points matches the target coreset size, we're done
        if len(indices) <= four_to_g:
            return indices
        else:
            # Partition the set input indices into four disjoint sets
            partition_set = np.array_split(indices,4)
            # Initialize an array to hold outputs of recursive calls
            compress_outputs = []
            for x in partition_set:
                # Recursively call compress on the four subsets and
                # add its output to the list
                compress_output = _compress(x)
                compress_outputs.append(compress_output)
            # Merge outputs of the recursive calls to compress
            combined_compress_output = np.concatenate(compress_outputs)
            # Run halving on the combined output
            indices_into_combined = halve(X[combined_compress_output])
            # Return indices into the original input set X
            return combined_compress_output[indices_into_combined]

    return _compress(indices)

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

def largest_power_of_four(n):
    """Returns largest power of four less than or equal to n
    """
    return 4**( (n.bit_length() - 1 )//2)