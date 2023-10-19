"""Compress, Compress++, and Symmetrize.

Implementations of the Compress, Compress++, and Symmetrize metaprocedures of
  Abhishek Shetty, Raaz Dwivedi, and Lester Mackey.
  Distribution Compression in Near-linear Time.
  https://arxiv.org/pdf/2111.07941.pdf
"""

import numpy as np
from numpy import zeros, where
from numpy.random import default_rng, SeedSequence
import math
from goodpoints import compressc, kt

def compresspp_kt(X, kernel_type, k_params=np.ones(1), g=0, num_bins=4,
                  delta=0.5, seed=None, mean0=False):
    """Returns coreset of size sqrt(n') as row indices into X. 
    Here n' = the largest power of 4 less than or equal to n. 
    The coreset is obtained by dividing the rows of X into num_bins 
    consecutive bins; running Compress(g) on each bin with the specified 
    kernel and Halve = symmetrized target kernel thinning; and running 
    target kernel thinning on the concatenated per-bin coresets.

    Note: When n is not a power of 4, the code reduces the index 
      count to n' by standard thinning (i.e., by selecting every t-th index) 
      backward from the end prior to running Compress.

    Args:
      X: Input sequence of sample points with shape (n, d)
      kernel_type: Byte string name of kernel to use:
        b"gaussian" for sum-of-Gaussians kernel 
          sum_j exp(-||x-y||_2^2/k_params[j]);
        b"sobolev" for sum-of-Sobolevs kernel
          sum_j Sobolev(x, y, smoothness = k_params[j])
      k_params: Double array of kernel parameters
      g: Oversampling parameter, a nonnegative integer
      num_bins: Number of bins, a power of 4 <= n
      delta: Failure probability parameter for kernel thinning
      seed: Nonnegative integer seed to initialize a random number 
        generator or None to set no seed
      mean0: If False, final KT call minimizes MMD to empirical measure over 
        the input points. Otherwise minimizes MMD to the 0 measure; this
        is useful when the kernel has expectation zero under a target measure.
    """
    # If the n is not a power of 4, thin down to the nearest 
    # power of 4 using standard thinning (i.e., by retaining every t-th index)
    n = X.shape[0]
    nearest_pow_four = largest_power_of_four(n)
    if nearest_pow_four != n:
        # Thin backwards from the end
        input_indices = np.linspace(n-1, 0, nearest_pow_four, dtype=int)[::-1]
        return input_indices[ compresspp_kt(
            X[input_indices], kernel_type, k_params=k_params, g=g, 
            num_bins=num_bins, delta=delta, seed=seed, mean0=mean0) ]
    
    # Record target number of halving rounds performed by Thin
    # Note: (num_bins.bit_length() - 1)//2 = log2(sqrt(num_bins))
    m = g + (num_bins.bit_length() - 1)//2
    
    # If 2^m >= sqrt(n) or equivalently 0 >= log2(sqrt(n)) - m, then 
    # no Compress step is needed: Thin directly to size sqrt(n)
    # Note: (n.bit_length() - 1)//2 = log2(sqrt(n))
    log2_sqrtn = (n.bit_length()-1)//2
    if log2_sqrtn <= m:
        # Allocate memory for kernel matrix
        K = np.empty((n, n))
        # Compute compress_coreset kernel matrix in place
        compressc.compute_K(X, np.arange(n, dtype=int), kernel_type, k_params, 
                            K)
        return kt.thin_K(K, K, log2_sqrtn, delta=delta, seed=seed, mean0=mean0)
        
    # Otherwise, divide failure probability between Compress and Thin rounds
    thin_frac = m / (m + (2**m) * (log2_sqrtn - m))
    thin_delta = delta * thin_frac
    compress_delta = delta * (1-thin_frac)
    
    # Generate one seed for the Compress step and one for the final Thin step
    seed_seqs = SeedSequence(seed).spawn(2)
    compress_seed = seed_seqs[0].generate_state(1)
    thin_seed = seed_seqs[1].generate_state(1)
    
    #
    # Compress step
    #
    # Break input into num_bins bins, use Compress(g) to create a coreset of size 
    # 2^g sqrt(n / num_bins) for each bin, and concatenate bin coresets
    compress_coreset = compress_kt(
        X, kernel_type, g=g, num_bins=num_bins, k_params=k_params, 
        delta=compress_delta, seed=compress_seed)
    
    #
    # Thin step
    #
    # Allocate memory for kernel matrix
    compress_coreset_size = compress_coreset.shape[0]
    K = np.empty((compress_coreset_size, compress_coreset_size))
    # Compute compress_coreset kernel matrix in place
    compressc.compute_K(X, compress_coreset, kernel_type, k_params, K)
    # Use target kt.thin to reduce coreset size from 2^g * sqrt(n * num_bins)
    # to sqrt(n)
    return compress_coreset[ 
        kt.thin_K(K, K, m, delta=thin_delta, seed=thin_seed, mean0=mean0)]

def compress_kt(X, kernel_type, k_params=np.ones(1), g=0, num_bins=1, 
                delta=0.5, seed=None):
    """Returns coreset of size min(n', 2^g sqrt(n' * num_bins)) as row indices 
    into X. Here n' = num_bins times the largest power of 4 less than or equal 
    to n / num_bins. The coreset is obtained by dividing the rows of X into 
    num_bins consecutive bins, running Compress(g) on each bin with the 
    specified kernel and Halve = symmetrized target kernel thinning, and 
    concatenating the resulting per-bin coresets.

    Note: When n / num_bins is not a power of 4, the code reduces the index 
      count to n' by standard thinning (i.e., by selecting every t-th index) 
      backward from the end prior to running Compress.
        
    Args:
      X: Input sequence of sample points with shape (n, d)
      kernel_type: Byte string name of kernel to use:
        b"gaussian" for sum-of-Gaussians kernel 
          sum_j exp(-||x-y||_2^2/k_params[j]);
        b"sobolev" for sum-of-Sobolevs kernel
          sum_j Sobolev(x, y, smoothness = k_params[j])
      k_params: Double array of kernel parameters
      g: Oversampling parameter, a nonnegative integer
      num_bins: Number of bins, a positive integer <= n
      lam_sqd: Double array of squared Gaussian kernel bandwidths to compute
        the sum-of-Gaussians kernel k(x,y) = sum_j exp(-||x-y||_2^2/lam_sqd[j])
      delta: Failure probability parameter for kernel thinning
      seed: Nonnegative integer seed to initialize a random number 
        generator or None to set no seed
    """
    # If the n // num_bins is not a power of 4, thin down to the nearest 
    # power of 4 using standard thinning (i.e., by retaining every t-th index)
    n = X.shape[0]
    n_per_bin = n // num_bins
    nearest_pow_four = largest_power_of_four(n_per_bin)
    if nearest_pow_four != n_per_bin:
        # Thin backwards from the end
        new_n = nearest_pow_four*num_bins
        input_indices = np.linspace(n-1, 0, new_n, dtype=int)[::-1]
        X = X[input_indices]
        n = new_n
    else:
        input_indices = None
    
    # Allocate array for storing coreset indices
    coreset_size = min(n, int(np.sqrt(n * num_bins)*(2**g)))    
    output_indices = np.empty(coreset_size, dtype=np.int64)
    
    # Run Compress on each bin
    if np.isscalar(k_params):
        k_params = np.array([k_params], dtype=np.double)
    if seed is None:
        # Pass negative seed value to indicate no seed should be set
        seed = -1
    compressc.compress(X, g, num_bins, kernel_type, k_params, delta, seed, 
                       output_indices)
    if input_indices is None:
        return output_indices
    else:
        # Return indices into the original matrix X
        return input_indices[output_indices]
     
def compress(X, halve, g = 0, indices = None):
    """Returns Compress coreset of size 2^g sqrt(n') as row indices into X.
    Here, n' is the largest power of 4 less than or equal to n (if indices 
    is None) or to len(indices) (if indices is not None).

    Note: When the number of indices to be compressed is not a power of 4, 
        the code reduces the index count to n' using standard thinning 
        (i.e., by selecting every t-th index) starting from the end
        prior to running Compress.

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
    # If no indices specified, set to range(n)
    if indices is None:
        indices = np.arange(X.shape[0], dtype=int)

    # If the number of indices is not a power of 4, thin down to the nearest 
    # power of 4 using standard thinning (i.e., by retaining every t-th index)
    n = len(indices)
    nearest_pow_four = largest_power_of_four(n)
    if nearest_pow_four != n:
        # Thin backwards from the end
        indices = indices[np.linspace(n-1, 0, nearest_pow_four, dtype=int)[::-1]]

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
        thin: Function that takes in an (2^g sqrt(n'), d) numpy array Y and returns
          sqrt(n') row indices into Y, identifying a thinned coreset
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
        rng = default_rng(seed)
        # create a vector of boolean values
        n = len(X)
        mask = zeros(n, dtype=bool)
        # set mask values of halve output to one
        mask[halve_output] = True
        if rng.choice([-1, 1]) == 1: # with probability half
            # take either mask or negation of mask
            mask = ~mask
        # return indices corresponding to mask 
        return where(mask)[0] 
    return symmetrized_halve

def largest_power_of_four(n):
    """Returns largest power of four less than or equal to n
    """
    return 4**( (n.bit_length() - 1 )//2)