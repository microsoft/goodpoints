"""Compress functionality.

Cython implementation of the Compress algorithm of
  Abhishek Shetty, Raaz Dwivedi, and Lester Mackey.
  Distribution Compression in Near-linear Time.
  https://arxiv.org/pdf/2111.07941.pdf
  
with symmetrized target kernel halving.
"""

import numpy as np
cimport numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport cython
from libc.math cimport exp, log2
# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. 
np.import_array()
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp
from cpython.pycapsule cimport PyCapsule_GetPointer
from numpy.random import PCG64, Generator
from numpy.random cimport bitgen_t
from goodpoints.ktc cimport thin_K
from goodpoints.sobolevc cimport (sobolev_kernel_two_points,
                                  sobolev_kernel_one_point)
from goodpoints.gaussianc cimport (gaussian_kernel_two_points, 
                                  gaussian_kernel_one_point)
'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Compress Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

# Define function type for two point kernel call with auxiliary
# parameter array s, e.g., (x,y,s) -> sum_l k_{s[l]}(x,y)
ctypedef double (*kernel)(const double[:], const double[:], 
                          const double[:]) noexcept nogil

# Define function type for one point kernel diagonal call with 
# auxiliary parameter array s, e.g., (x,s) -> sum_l k_{s[l]}(x,x)
ctypedef double (*kernel_diag)(const double[:], const double[:]) noexcept nogil

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void compute_K(const double[:, :] X, 
                     const long[:] input_indices, 
                     const char* kernel_type, 
                     const double[:] k_params,
                     double[:, :] K) noexcept nogil:
    """
    Computes kernel matrix for points X[input_indices] and stores in K
    
    Args:
      X: array of size (N, d)
      input_indices: array of size n <= N
      kernel_type: Byte string name of kernel to use:
        b"gaussian" for (sum of) Gaussian kernels with squared bandwidth params,
        b"sobolev" for (sum of) Sobolev kernels with smoothness params
      k_params: array of kernel parameters (e.g., squared bandwidths for sum of
        Gaussian kernels or smoothness parameters for sum of Sobolev)
      K: array of size (n, n) for storing the kernel matrix
    """
    # Select kernel functions to match kernel type
    cdef kernel k
    cdef kernel_diag kdiag
    # strcmp returns 0 when two const char* strings are equal
    # b prefix designates byte string
    if strcmp(kernel_type, b"gaussian") == 0:
        k = gaussian_kernel_two_points 
        kdiag = gaussian_kernel_one_point
    elif strcmp(kernel_type, b"sobolev") == 0:
        k = sobolev_kernel_two_points 
        kdiag = sobolev_kernel_one_point
    # Populate kernel matrix using these kernel functions
    _compute_K(X, input_indices, k, kdiag, k_params, K)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef void _compute_K(const double[:, :] X, 
                     const long[:] input_indices, 
                     const kernel k,
                     const kernel_diag kdiag, 
                     const double[:] k_params,
                     double[:, :] K) noexcept nogil:
    """
    Computes kernel matrix for points X[input_indices] and stores in K
    
    Args:
      X: array of size (N, d)
      input_indices: array of size n <= N
      k: kernel function k(x,y,s) with auxiliary parameter array s
      kdiag: kernel diagonal function; kdiag(x, s) = k(x, x, s)
      k_params: array of kernel parameters (e.g., squared bandwidths for sum of
        Gaussian kernels or smoothness parameters for sum of Sobolev)
      K: array of size (n, n) for storing the kernel matrix
    """
    
    cdef long i, j
    cdef long n = input_indices.shape[0]
    cdef const double[:] X_input_i, X_input_j
    cdef double[:] K_i
    cdef double k_val
    for i in range(n):
        X_input_i = X[input_indices[i]]
        K_i = K[i]
        K_i[i] = kdiag(X_input_i, k_params) 
        j = 0 
        while j < i: 
            X_input_j = X[input_indices[j]]
            k_val = k(X_input_i, X_input_j, k_params) 
            K_i[j] = k_val
            K[j, i] = k_val
            j += 1

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef void halve(const double[:,:] X,
                const long[:] input_indices,
                bitgen_t* rng,
                const kernel k,
                const kernel_diag kdiag, 
                const double[:] k_params,
                const double halve_prob,
                double[:,:] K,
                double[:] aux_double_mem,
                long[:,:] aux_long_mem,
                long[:] output_indices) noexcept nogil:
    """
    Return halved coreset for X[input_indices] with the halved indices into X stored 
    in output_indices. Halving is performed by symmetrized kt.thin with parameter
    delta and kernel k.
    
    Requirement: the length n of input_indices must be even
    
    Args:
      X: array of size (N, d)
      input_indices: array of size n <= N
      rng: pointer to random number generator
      k: kernel function k(x,y,s) with auxiliary parameter array s
      kdiag: kernel diagonal function; kdiag(x, s) = k(x, x, s)
      k_params: array of kernel parameters (e.g., squared bandwidths for sum of
        Gaussian kernels or smoothness parameters for sum of Sobolev)
      halve_prob: Runs thin_K with failure probability parameter 
        delta = halve_prob * n^2
      K: shape (2*n_out, 2*n_out) array for storing a kernel matrix; will be
        modified in place
      aux_double_mem: scratch space of length 4*n_out; will be modified in place
      aux_long_mem: scratch space of shape (2, n_out); will be modified in place
      output_indices: array of size n//2; will be modified in place store the output row indices into X
        representing coreset; 
    """
    
    # Populate kernel matrix
    _compute_K(X, input_indices, k, kdiag, k_params, K)
    
    # Failure probability parameter accepted by thin_K
    cdef double delta = halve_prob * (input_indices.shape[0] ** 2)
    
    # Run kernel thinning to select half of indices; 
    # Note: at this stage, output_indices will contain 
    # indices into the input_indices array, and not into X
    cdef bint unique = True
    cdef bint mean0 = False
    thin_K(K, rng, delta, unique, mean0, 
           aux_double_mem, aux_long_mem, output_indices)

    # Return the selected coreset as indices into X
    cdef long coreset_size = output_indices.shape[0]
    cdef long i
    for i in range(coreset_size):
        output_indices[i] = input_indices[output_indices[i]]

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef void _compress(const double[:, :] X, 
                    const long[:] input_indices,
                    const long base_size,
                    const kernel k,
                    const kernel_diag kdiag, 
                    const double[:] k_params,
                    const double halve_prob,
                    bitgen_t* halve_rng,
                    double[:,:] K,
                    double[:] aux_double_mem,
                    long[:,:] aux_long_mem,
                    long[:] intermediate_indices, 
                    long[:] output_indices) noexcept nogil: 
    """Forms Compress coreset for X[input_indices] of length 
    n_out = sqrt(base_size * len(input_indices))/2 representing
    row indices into X.  Stores coreset in output_indices
    
    Note: Assumes n is a power of 4, that n >= base_size 

    Args:
      X: Input sequence of sample points with shape (N, d)
      input_indices: array of size n <= N
      base_size: Size at which Compress amounts to directly calling
        Halve on the input points without further recursive calls to Compress;
        corresponds to 4^{g+1} for integer g >= 0 the oversampling parameter 
        of Compress
      k: kernel function k(x,y,s) with auxiliary parameter array s
      kdiag: kernel diagonal function; kdiag(x, s) = k(x, x, s)
      k_params: array of kernel parameters (e.g., squared bandwidths for sum of
        Gaussian kernels or smoothness parameters for sum of Sobolev)
      halve_prob: Halving probability parameter accepted by halve
      halve_seed: Pointer to an RNG for halving algorithm
      K: array for storing a kernel matrix; will be modified in place;
        has shape (2*n_out, 2*n_out) 
      aux_double_mem: scratch space of; will be modified in place;
        has length 4*n_out 
      aux_long_mem: scratch space; will be modified in place;
        has shape (2, n_out) 
      intermediate_indices: array for storing indices returned by recursive 
        calls to compress; will be modified in place; has length 4*n_out 
      output_indices: array for storing the output coreset indexing the 
        rows of X; will be modified in place; has length n_out 
    """
    # Number of input points
    cdef long n = input_indices.shape[0]

    # If input size equals base_size, call Halve directly on input indices
    if n == base_size:
        # Call Halve directly on the input_indices
        halve(X, input_indices, halve_rng, k, kdiag, k_params, halve_prob, 
              K, aux_double_mem, aux_long_mem, output_indices)
        return

    # Divide input indices into four blocks, call compress on each
    # block, and store result in corresponding block of
    # intermediate_indices
    cdef long n_out = output_indices.shape[0]
    cdef long child_in_len = n//4
    cdef long child_out_len = n_out//2
    cdef long child_in_start = 0
    cdef long child_out_start = 0
    
    # Pass the same subset of the auxiliary memory to each child
    cdef long halve_size = 2*n_out
    cdef double[:,:] child_K = K[:n_out,:n_out]
    cdef double[:] child_aux_double_mem = aux_double_mem[:halve_size]
    cdef long[:,:] child_aux_long_mem = aux_long_mem[:,:child_out_len]
    cdef long[:] child_intermediate_indices, child_output_indices
    # Use the first half of intermediate_indices to store childrens'
    # output indices
    child_output_indices = intermediate_indices[:halve_size]
    # Use the second half of intermediate_indices to store child's intermediate
    # indices
    child_intermediate_indices = intermediate_indices[halve_size:]
    
    # Recursively _compress on each child bin
    cdef long i    
    for i in range(4):
        _compress(
            X, 
            input_indices[child_in_start:child_in_start+child_in_len],
            base_size, k, kdiag, k_params, halve_prob, 
            halve_rng,
            child_K,
            child_aux_double_mem,
            child_aux_long_mem,
            child_intermediate_indices,
            child_output_indices[child_out_start:child_out_start+child_out_len]
           )
        child_in_start += child_in_len
        child_out_start += child_out_len
    
    # Run halving on child_output_indices
    halve(X, child_output_indices, halve_rng, k, kdiag, k_params, halve_prob, 
          K, aux_double_mem, aux_long_mem, output_indices)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void compress(const double[:, :] X, 
                    const unsigned long g, 
                    const long num_bins,
                    const char* kernel_type, 
                    const double[:] k_params,
                    const double delta,
                    const long halve_seed,
                    long[:] output_indices) noexcept nogil:
    """Partitions rows of X consecutively into num_bins bins,
    forms Compress(g) coreset for each bin as row indices into X, and 
    stores concatenated coresets in output_indices.
    
    Note: Assumes n = num_bins times a power of 4
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      g: Oversampling parameter, a nonnegative integer
      num_bins: Number of bins
      kernel_type: Byte string name of kernel to use:
        b"gaussian" for (sum of) Gaussian kernels with squared bandwidth params,
        b"sobolev" for (sum of) Sobolev kernels with smoothness params
      k_params: Array of kernel parameters
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      halve_seed: Nonnegative integer seed to initialize a random number 
        generator or a negative integer indicating no seed should be set
      output_indices: Array for storing the output coreset indexing the rows 
        of X; will be modified in place; has length min(n_out, n)
        for n_out = 2^g *sqrt(n num_bins)  
    """ 
    
    # Number of sample points
    cdef long n = X.shape[0]
    # Number of input points per bin
    cdef long bin_size = n // num_bins
    
    # Smallest input size for which Compress(g) output coreset 
    # is strictly smaller than input
    cdef long base_size = 4**(g+1)
    # If input size is smaller than base_size, return all input indices.
    # Under our assumption of n = num_bins times a power of 4, this is
    # equivalent to saying return all input indices whenever the target
    # output size >= n 
    cdef long i
    if bin_size < base_size:
        for i in range(n):
            output_indices[i] = i
        return

    # Size of the coreset outputted for each bin
    cdef long bin_out_size = output_indices.shape[0] // num_bins
    # Allocate auxiliary memory for largest instance of Halve
    cdef long max_halve_size = 2 * bin_out_size
    cdef double *K_ptr = <double *> malloc(max_halve_size*max_halve_size * sizeof(double))
    cdef double[:, :] K
    cdef double *aux_double_ptr = <double *> malloc(2*max_halve_size * sizeof(double))
    cdef double[:] aux_double_mem
    cdef long *aux_long_ptr = <long *> malloc(2 * bin_out_size * sizeof(long))
    cdef long[:,:] aux_long_mem
    cdef long *intermediate_indices_ptr = <long *> malloc(2*max_halve_size* sizeof(long))
    cdef long[:] intermediate_indices 
    # Create array of input indices to be halved
    cdef long *input_indices_ptr = <long *> malloc(bin_size * sizeof(long))
    cdef long[:] input_indices
    # Instantiate random number generator for Halve algorithm
    cdef bitgen_t *halve_rng    
    cdef const char *capsule_name = "BitGenerator" 
    with gil:
        # Cast pointers to memoryviews
        K = <double[:max_halve_size, :max_halve_size]>K_ptr
        aux_double_mem = <double[:2*max_halve_size]>aux_double_ptr
        aux_long_mem = <long[:2,:bin_out_size]>aux_long_ptr
        intermediate_indices = <long[:2*max_halve_size]>intermediate_indices_ptr
        input_indices = <long[:bin_size]>input_indices_ptr

        # Store the generator in a variable before extracting capsule
        # to avoid the issue of generator being freed prematurely
        if halve_seed >= 0:
            halve_generator = Generator(PCG64(halve_seed)) # PyObject
        else:
            # Do not set seed if halve_seed is negative
            halve_generator = Generator(PCG64()) # PyObject
        halve_capsule = halve_generator.bit_generator.capsule # PyObject
        halve_rng = <bitgen_t *> PyCapsule_GetPointer(halve_capsule, capsule_name)
    
    # Compute base halving probability used by kernel thinning
    halve_prob = delta / base_size / (log2(bin_size)/2 - g) / n

    # Select kernel functions to match kernel type
    cdef kernel k
    cdef kernel_diag kdiag
    # strcmp returns 0 when two const char* strings are equal
    # b prefix designates byte string
    if strcmp(kernel_type, b"gaussian") == 0:
        k = gaussian_kernel_two_points 
        kdiag = gaussian_kernel_one_point
    elif strcmp(kernel_type, b"sobolev") == 0:
        k = sobolev_kernel_two_points 
        kdiag = sobolev_kernel_one_point
    
    # Compress each bin
    # Keep track of current bin's starting index into output indices
    # and into input rows of X
    cdef long bin_out_start = 0, bin_in_start = 0
    # Output size for each bin
    cdef long j
    for j in range(num_bins):
        # Specify input indices for this bin
        for i in range(bin_size):
            input_indices[i] = bin_in_start + i
         
        # Run compress algorithm to populate output indices
        _compress(X, input_indices, base_size, 
                  k, kdiag, k_params, halve_prob, 
                  halve_rng, 
                  K, aux_double_mem, aux_long_mem,
                  intermediate_indices,
                  output_indices[bin_out_start:bin_out_start+bin_out_size]
                 )
        
        bin_out_start += bin_out_size
        bin_in_start += bin_size
    
    # Free allocated memory
    free(K_ptr)
    free(aux_double_ptr)
    free(aux_long_ptr)
    free(intermediate_indices_ptr)
    free(input_indices_ptr)