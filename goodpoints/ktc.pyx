"""Kernel thinning functionality.

Cython implementations of the kernel halving, best, and refine algorithms of
  Raaz Dwivedi and Lester Mackey.
  Kernel Thinning.
  https://arxiv.org/pdf/2105.05842.pdf
  
the target kernel thinning (with m = 1) algorithm of
  Raaz Dwivedi and Lester Mackey.
  Generalized Kernel Thinning.
  https://arxiv.org/pdf/2110.01593.pdf
  
and the optimal four-point halving algorithm of 
  Carles Domingo-Enrich, Raaz Dwivedi, and Lester Mackey.
  Compress Then Test: Powerful Kernel Testing in Near-linear Time.
  https://arxiv.org/pdf/2301.05974.pdf
"""

import numpy as np
cimport numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport cython
from libc.math cimport sqrt, log
from libc.stdio cimport printf
# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. 
np.import_array()
from numpy.random cimport bitgen_t
from numpy.random.c_distributions cimport (random_standard_uniform,
      random_standard_uniform_fill)
from libc.stdlib cimport malloc, free


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kernel Thinning Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

@cython.boundscheck(False) # turn off bounds-checking for this function  
@cython.wraparound(False)  # turn off negative index wrapping for this function  
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef void thin_K(const double[:, :] K,
                  bitgen_t* rng,
                  const double delta,
                  const bint unique,
                  const bint mean0,
                  double[:] aux_double_mem,
                  long[:,:] aux_long_mem,
                  long[:] output_indices) noexcept nogil:
    
    """Produces a kt.thin(m = 1) coreset of size n//2 given
    a size n x n kernel matrix K and stores it in output_indices.
    
    Note: Assumes that n is even.
    Note: Returned coreset indices need not be sorted.
    
    Args:
      K: Matrix of KT-SPLIT kernel evaluations with shape (n, n)
      rng: random number generator
      delta: Runs halve_K with failure probability parameter delta
      unique: If True, constrains the output to never contain the same row index 
        more than once and symmetrizes output by returning the chosen coreset
        or its complement with equal probability
      mean0: If False, the KT-SWAP stage will minimize MMD to the empirical 
        distribution Pn over the n input points by (implicitly) recentering 
        the kernel matrix with respect to Pn.  If True, no recentering is 
        performed. This is useful when the kernel is already centered with 
        respect to a target distribution P as then KT-SWAP will minimize 
        MMD to P.
      aux_double_mem: scratch space array of size 2n; will be modified in place
      aux_long_mem: scratch space array of size (2, n//2); will be modified in place
      output_indices: array of size n//2, would store the output row indices into K
       representing coreset; will be modified in place
    """
    cdef long n = K.shape[0]
    
    if n == 4:
        # Use optimized implementation when n == 4
        if mean0:
            opt_halve4_mean0_K(K, output_indices)
        else:
            opt_halve4_K(K,
                         random_standard_uniform(rng),
                         output_indices) 
        return

    ########## halve_K ##########
    cdef long i, j
    cdef long coreset_size = n//2  
    
    # Generate uniform random numbers to be used by halve_K
    cdef double[:] uniforms = aux_double_mem[:coreset_size]
    random_standard_uniform_fill(rng, coreset_size, &uniforms[0])
        
    # Divide input into two coresets, stored in coresets
    cdef long[:, :] coresets = aux_long_mem
    halve_K(K, delta, uniforms, coresets)
    
    # Treat coresets[0] as the selected coreset and coresets[1] as its 
    # non-coreset complement
    # Note: non-coreset is the complement of coreset only when n is even
    cdef long[:] coreset = coresets[0]
    cdef long[:] non_coreset = coresets[1]

    ########## best_K ##########

    cdef double[:] meanK
    cdef const double[:] K_i
    if mean0:
        # No recentering is necessary: make meanK None
        meanK = None
    else:
        # Recentering is necessary: 
        # compute the row means of K (i.e., meanK = K.mean(axis=1))
        # for best_K and refine_K and store in first half of aux_double_mem
        meanK = aux_double_mem[:n]
        for i in range(n):
            meanK[i] = 0
            K_i = K[i]
            for j in range(n):
                meanK[i] += K_i[j]
            meanK[i] /= n
        
    # Store standard thinning coreset in output_indices
    # np.flip(np.arange(n-1,-1,-(n//coreset_size)))
    # = np.flip(np.arange(n-1,-1,-2))
    j = coreset_size-1
    i = n-1
    while j>=0:
        output_indices[j] = i
        i -= 2
        j -= 1
    
    # Select the better of halve_K coreset (in coresets[0])
    # and standard thinning coreset (in output_indices)
    # Check whether standard thinning coreset was selected
    # by examining whether coreset points to same memory as output indices
    if &(best_K(K, coresets, meanK, output_indices)[0]) == &output_indices[0]:
        # Select higher-quality standard thinning coreset as coreset
        coreset = output_indices
        # Store complement of coreset into non_coreset
        # Note: non-coreset is the complement of coreset only when n is even
        j = coreset_size-1
        i = n-2
        while j>=0:
            non_coreset[j] = i
            i -= 2
            j -= 1
    
    ########## refine_K ##########
    
    # Use the second half of aux_double_mem as space for sufficient statistics,
    # one per input point for ktc.refine_K
    cdef double[:] sufficient_stat = aux_double_mem[n:]
   
    # Refine quality of coreset
    refine_K(K, coreset, meanK, 
             sufficient_stat, unique, 
             non_coreset)
    
    ########## symmetrize ##########
    # Flip a fair coin to decide whether to return coreset or non_coreset
    if random_standard_uniform(rng) > 0.5:
        coreset = non_coreset
    
    # Copy selected coreset into output_indices
    if &coreset[0] != &output_indices[0]:
        for i in range(coreset_size):
            output_indices[i] = coreset[i]


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kernel Halving Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void opt_halve4_K(const double[:,:] K,
                        const double uniform,
                        long[:] coreset) noexcept nogil:
    """Identifies a two-point coreset with smallest MMD to the four-point input point set X.
    If uniform < .5, stores those point indices in coreset; otherwise, stores the complement of  
    those indices in coreset. Here X is implicitly represented by its kernel matrix K satisfying 
    K[ii,:] = kernel(X[ii], X).
    
    Args:
      K: Matrix of pairwise kernel evaluations with shape (4, 4)
      uniform: Random real number distributed uniformly on the unit interval [0,1]
      coreset: Preallocated 1D array with length 2 representing the coreset of row indices into K
        to be returned; will be modified in place
    """
    cdef double K12_plus_K03 = K[1,2] + K[0,3]
    cdef double K01_plus_K23 = K[0,1] + K[2,3]
    cdef double K02_plus_K13 = K[0,2] + K[1,3]

    if K12_plus_K03 < K01_plus_K23:
        if K12_plus_K03 < K02_plus_K13:
            # MMD^2 of coreset [0,3] = MMD^2 of coreset [1,2] 
            # = const - (K02_plus_K13 + K01_plus_K23) / 4
            if uniform < .5:
                coreset[0] = 0; coreset[1] = 3
            else:
                coreset[0] = 1; coreset[1] = 2
        else: 
            # MMD^2 of coreset [0,2] = MMD^2 of coreset [1,3] 
            # = const - (K12_plus_K03 + K01_plus_K23) / 4
            if uniform < .5:
                coreset[0] = 0; coreset[1] = 2
            else:
                coreset[0] = 1; coreset[1] = 3               
    elif K01_plus_K23 < K02_plus_K13:
        # MMD^2 of coreset [0,1] = MMD^2 of coreset [2,3]
        # = const - (K02_plus_K13 + K12_plus_K03) / 4
        if uniform < .5:
            coreset[0] = 0; coreset[1] = 1
        else:
            coreset[0] = 2; coreset[1] = 3                            
    else:
        # MMD^2 of coreset [0,2] = MMD^2 of coreset [1,3] 
        # = const - (K12_plus_K03 + K01_plus_K23) / 4
        if uniform < .5:
            coreset[0] = 0; coreset[1] = 2
        else:
            coreset[0] = 1; coreset[1] = 3 

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void opt_halve4_mean0_K(const double[:,:] K,
                              long[:] coreset) noexcept nogil:
    """Identifies a two-point coreset with smallest MMD to the zero measure.
    Here X is implicitly represented by its kernel matrix K satisfying 
    K[ii,:] = kernel(X[ii], X).
    
    Args:
      K: Matrix of pairwise kernel evaluations with shape (4, 4)
      coreset: Preallocated 1D array with length 2 representing the coreset of 
        row indices into K to be returned; will be modified in place
    """
    cdef long i, j
    cdef double sqd_mmd, best_sqd_mmd, diagK_j

    # Start with coreset (0,1)
    coreset[0] = 0
    coreset[1] = 1
    best_sqd_mmd = K[0,0] + K[1,1] + 2*K[0,1]
    # Check if any other coreset has smaller sqd_mmd to 0 measure
    for j in range(2,4):
        diagK_j = K[j,j]
        for i in range(0,j):
            sqd_mmd = diagK_j + K[i,i] + 2*K[i,j]
            if sqd_mmd < best_sqd_mmd:
                best_sqd_mmd = sqd_mmd
                coreset[0] = i; coreset[1] = j


@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void halve_K(const double[:,:] K,
                   const double delta,
                   const double[:] uniforms,
                   long[:,:] coresets,
                   const long[:] input_indices = None) noexcept nogil:    
    """Identifies two KT-SPLIT coresets of size floor(n/2) and stores them 
    in coresets. If input_indices is None, partitions the n row indices of K.
    Otherwise, partitions the n indices indicated in input_indices.
    
    Args:
      K: 2D array of KT-SPLIT kernel evaluations 
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      uniforms: Array of at least floor(n/2) independent random variables 
        uniformly distributed on the unit interval [0,1]
      coresets: Array of shape (2, floor(n/2)) for storing KT-SPLIT coresets; will be
        modified in place
      input_indices: None or array of indices of length n indexing the rows and 
        columns of K
    """
    cdef bint no_indices = input_indices is None
    
    cdef long n
    if no_indices:
        n = K.shape[0]
    else:
        n = input_indices.shape[0]
        
    cdef long num_points_in_coreset = n//2
    
    cdef double sig_sqd = 0
    cdef double sig_sqd_update
    cdef double log_multiplier = 2*log(2*n/delta)
    
    cdef long i, point0, point1
    cdef double b_sqd, thresh, alpha, prob_point1
    cdef long j, c1j, c0j
    for i in range(num_points_in_coreset):
        if no_indices:
            point0 = 2*i
            point1 = 2*i+1
        else:
            point0 = input_indices[2*i]
            point1 = input_indices[2*i+1]
        # Compute b^2 = ||f||^2 = ||k(point0,.) - k(point1,.)||_k^2
        b_sqd = K[point0,point0] - 2*K[point0,point1] + K[point1,point1] 
        # Compute adaptive failure threshold
        # thresh = max(b sig sqrt(log_multiplier), b^2)
        thresh = sqrt(sig_sqd*b_sqd*log_multiplier)
        if b_sqd > thresh:
            thresh = b_sqd
        # Update sub-Gaussian parameter
        if sig_sqd == 0:
            sig_sqd = b_sqd
        elif thresh != 0:
            # Note: If threshold is zero, b_sqd is zero so sig_sqd does not change
            # If thresh != 0, update subGaussian parameter
            # s^2 += 2*b^2*(.5 + (b^2/(2 a) - 1)*s^2/a)_+ or equivalently
            # s^2 += b^2*(1 + (b^2/a - 2)*s^2/a)_+ 
            sig_sqd_update = 1+(b_sqd/thresh-2)*sig_sqd/thresh
            if sig_sqd_update > 0:
                sig_sqd += b_sqd*sig_sqd_update
        # To avoid division by zero, set zero threshold to arbitrary positive value
        # (Does not impact algorithm correctness as b_sqd = 0 as well)
        if thresh == 0: thresh = 1.
        
        # Compute alpha = sum(K[point0,coreset1] - K[point0,coreset0] + 
        #   K[point1,coreset0] - K[point1,coreset1])
        alpha = 0
        for j in range(i):
            c1j = coresets[1,j]
            c0j = coresets[0,j]
            alpha += K[point0,c1j] - K[point0,c0j] + K[point1,c0j] - K[point1,c1j]
        
        prob_point1 = 0.5*(1-alpha/thresh)
        if uniforms[i] <= prob_point1:
            coresets[0,i] = point1
            coresets[1,i] = point0
        else:
            coresets[0,i] = point0
            coresets[1,i] = point1
            
'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KT-SPLIT Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void split_K(const double[:,:] K,
                   const double delta,
                   const double[:,:] uniforms,
                   long[:,:] coresets) noexcept nogil:
    """Identifies 2^m KT-SPLIT coresets of size floor(n/2^m) and stores them 
    in coresets.
    
    Args:
      K: Matrix of KT-SPLIT kernel evaluations with shape (n, n)
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      uniforms: Array of shape (m, floor(n/2)) of independent random variables 
        uniformly distributed on the unit interval [0,1]
      coresets: Array of shape (2^m, floor(n/2^m)) for storing KT-SPLIT coresets; will be
        modified in place
    """
    cdef long m = uniforms.shape[0]    
    if m == 1:
        # Use streamlined kernel halving implementation
        halve_K(K, delta, uniforms[0], coresets)
        return
    cdef long i
    if m == 0:
        # Return range(K.shape[0]) as single coreset
        for i in range(K.shape[0]):
            coresets[0,i] = i
        return

    # Record the total number of elements in the output coresets array
    # We refer to this as n even though it is 2^m floor(n/2^m) in the notation
    # of the function header comment
    cdef long n = coresets.shape[0]*coresets.shape[1]

    
    # Allocate auxiliary memory for storing input indices
    cdef long *input_ptr = <long *> malloc(n * sizeof(long))
    # Obtain pointer to coresets memory
    cdef long *output_ptr = &coresets[0,0]

    # Maintain input coreset memoryview
    cdef long [:,:] input_coresets
    with gil:        
        if m % 2 == 0:
            # When m is even, point input to output_ptr memory initially
            input_coresets = <long[:1, :n]>output_ptr
            # Then point output_ptr to input memory
            output_ptr = input_ptr
        else: 
            # When m is odd, point input to input_ptr initially
            input_coresets = <long[:1, :n]>input_ptr
    
    # Initialize input_coresets to range(n)
    for i in range(n):
        input_coresets[0,i] = i
    
    # Maintain auxiliary memoryview for output coresets
    cdef long [:,:] output_coresets
        
    cdef long num_input_coresets = 1, output_coreset_size = n // 2
    cdef long uniforms_j_start, output_start
    cdef const double[:] uniforms_j
    # For each halving round
    cdef long j, l
    for j in range(m):
        if j == m-1:
            # Final halving round: use provided coresets array
            output_coresets = coresets
        else:
            # Cast output pointer to memoryview with size (2^{j+1}, n//2^{j+1}) 
            with gil:
                output_coresets = <long[:num_input_coresets*2, :output_coreset_size]>output_ptr

        # Partition each input coreset into halves using failure probability
        # delta * 2^j / m  and appropriate subset of uniform variables;
        # store results into associated output_coresets
        uniforms_j = uniforms[j]
        uniforms_j_start = 0
        output_start = 0
        for l in range(num_input_coresets):
            halve_K(K,  
                    delta * num_input_coresets / m, 
                    uniforms_j[uniforms_j_start:],
                    output_coresets[output_start:output_start+2],
                    input_coresets[l])
            uniforms_j_start += output_coreset_size
            output_start += 2
            
        # Use this round's output coresets as next round's input_indices
        output_ptr = &input_coresets[0,0]
        input_coresets = output_coresets
        # Update input / output sizes and counts
        num_input_coresets *= 2
        output_coreset_size = output_coreset_size // 2

    # Free allocated memory
    free(input_ptr)
    
'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KT Best Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef const long[:] best_K(const double[:,:] K,
                           const long[:,:] coresets,
                           const double[:] meanK,
                           const long[:] best_coreset) noexcept nogil:
    """After comparing with a baseline standard thinning coreset,
    selects the candidate coreset with smallest MMD to all input points in X
    (if meanK is not None) or to the zero measure (if meanK is None).
    Here X is implicitly represented by its kernel matrix K satisfying 
    K[ii,:] = kernel(X[ii], X).
    
    Args:
      K: Matrix of kernel evaluations with shape (n, n)
      coresets: 2D array with each row specifying the row indices of X belonging to a coreset
      meanK: None or array of length n with meanK[ii] = mean of K[ii,:]
      best_coreset: 1D array containing a coreset produced by standard thinning
    """
    cdef long num_coresets = coresets.shape[0]
    # Compute the relative MMD^2 of a standard thinning coreset
    # RelMMD^2 = MMD^2 - np.mean(K) = np.mean(K[coreset][:,coreset]) - 2*np.mean(K[coreset, :]))
    cdef double best_rel_mmd2 = squared_emp_rel_mmd_K(K, best_coreset, meanK)
    cdef double rel_mmd2
    cdef long i
    # Select the better of standard thinning coreset and the best input coreset
    if (num_coresets == 2) and (meanK is not None):
        # Only compare to first coreset, as the two coresets have equal MMD
        # when n is even
        rel_mmd2 = squared_emp_rel_mmd_K(K, coresets[0], meanK)
        if rel_mmd2 < best_rel_mmd2:
            best_coreset = coresets[0]
    else:
        for i in range(num_coresets):
            rel_mmd2 = squared_emp_rel_mmd_K(K, coresets[i], meanK)
            if rel_mmd2 < best_rel_mmd2:
                best_rel_mmd2 = rel_mmd2
                best_coreset = coresets[i]
    
    return(best_coreset)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double squared_emp_rel_mmd_K(const double[:,:] K,
                                  const long[:] coreset,
                                  const double[:] meanK) noexcept nogil:
    """Computes squared empirical relative MMD between a distribution weighted
    equally on all points and one weighted equally on the indices in coreset:
    RelMMD^2 = MMD^2 - np.mean(K) 
             = np.mean(K[coreset][:,coreset]) - 2*np.mean(K[coreset, :]))

    If meanK is None, then computes squared MMD between a distribution weighted
    equally on the indices in coreset and the zero measure:
    RelMMD^2 = np.mean(K[coreset][:,coreset]) 
    
    Args:
      K: Matrix of pairwise kernel evaluations with shape (n, n)
      coreset: Row indices of K representing coreset
      meanK: None or array of length n with meanK[ii] = mean of K[ii,:]
    """
    cdef long coreset_size = coreset.shape[0]
    cdef long sqd_coreset_size = coreset_size * coreset_size
    
    cdef double rel_mmd2 = 0
    cdef long i, j, cset_i
    if meanK is None:
        # Compute rel_mmd2 = np.mean(K[coreset][:,coreset])
        for i in range(coreset_size):
            cset_i = coreset[i]
            # Compute contribution of cset_i:
            # np.sum(K[coreset[i], coreset]) 
            for j in range(coreset_size):
                rel_mmd2 += K[cset_i,coreset[j]]
        # Normalize by number of entries
        return(rel_mmd2 / sqd_coreset_size)
    
    # Otherwise, compute  
    # rel_mmd2 = np.mean(K[coreset][:,coreset]) - 2*np.mean(meanK[coreset])
    cdef double K_cset_i_coreset_sum
    for i in range(coreset_size):
        cset_i = coreset[i]
        # Compute contribution of cset_i:
        # np.sum(K[coreset[i], coreset]) / sqd_coreset_size - 2 * meanK[cset_i] / coreset_size
        K_cset_i_coreset_sum = 0
        for j in range(coreset_size):
            K_cset_i_coreset_sum += K[cset_i,coreset[j]]
        rel_mmd2 += K_cset_i_coreset_sum / sqd_coreset_size - 2 * meanK[cset_i] / coreset_size    
    
    return(rel_mmd2)


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KT Refine Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void refine_K(const double[:,:] K,
                    long[:] coreset,
                    const double[:] meanK,
                    double[:] sufficient_stat,
                    const bint unique,
                    long[:] non_coreset) noexcept nogil:
    """
    Replaces each element of a coreset in turn by the input point that yields 
    the greatest decrease in MMD between the resulting coreset and all input 
    points (if meanK is not None) or between the resulting coreset and the 
    zero measure (if meanK is None).
    Here X is implicitly represented by its kernel matrix K satisfying 
    K[ii,:] = kernel(X[ii], X).
    
    Note: Returned coreset and non_coreset indices need not be sorted.
    
    Args:
      K: Matrix of kernel evaluations with shape (n, n)
      coreset: Row indices of K representing coreset; will be modified in place
      meanK: None or array of length n with meanK[ii] = mean of K[ii,:]
      sufficient_stat: Array of shape (n) for storing sufficient statistics; 
        will be modified in place
      unique: If True, constrains the output to never contain the same row index more than once
      non_coreset: If unique is True, this is an array of row indices of K representing points 
        not in coreset; will be modified in place and will contain the complement of coreset
        when this function returns. If unique is False, this argument is ignored.
    """
    
    cdef long n = K.shape[0]
    cdef long coreset_size = coreset.shape[0]
    cdef double two_over_coreset_size = 2./coreset_size
    
    # Compute sufficient statistic representing how much each point would 
    # change the quantity coreset_size * MMD^2(P,Q) if added into the coreset
    # with weight 1/coreset_size.
    # Since coreset_size * MMD^2(P,Q) = coreset_size * (PPk - 2QPK + QQK),
    # the impact of adding (1/coreset_size) delta_y to Q = 
    #   -2 delta_y PK + delta_y delta_y K / coreset_size + 2 delta_y Q K.
    # Statistic will later be adjusted to remove the influence of an 
    # eliminated point
    cdef long i, j
    cdef double meanK_i_coreset
    if meanK is None:
        # Target measure P is the zero measure, so Pk = 0
        for i in range(n):
            meanK_i_coreset = 0
            for j in range(coreset_size):
                meanK_i_coreset += K[i,coreset[j]]
            sufficient_stat[i] = (K[i,i] + 2*meanK_i_coreset) / coreset_size
    else:
        # Target measure P is the empirical measure over input points
        for i in range(n):
            sufficient_stat[i] = K[i,i]/coreset_size
            meanK_i_coreset = 0
            for j in range(coreset_size):
                meanK_i_coreset += K[i,coreset[j]]
            meanK_i_coreset /= coreset_size
            sufficient_stat[i] += 2*(meanK_i_coreset-meanK[i])
    
    cdef long idx_into_coreset = 0
    cdef long coreset_point = coreset[idx_into_coreset]
    cdef long best_idx_into_non_coreset
    cdef long best_point, non_coreset_point
    cdef double min_sufficient_stat, stat_i
    cdef long non_coreset_size
    if unique:
        non_coreset_size = non_coreset.shape[0]
    
    while True:
        # Remove the contribution of coreset point from the normalized coreset sum in sufficient stat: 
        # - 2 delta_x delta_y K / coreset_size
        for i in range(n):
            sufficient_stat[i] -= K[coreset_point,i]*two_over_coreset_size
            
        # Find valid candidate point that would reduce MMD the most
        if unique:
            # First consider swapping back in the swapped out coreset point
            best_idx_into_non_coreset = -1
            min_sufficient_stat = sufficient_stat[coreset_point]
            # Next consider swapping in each non_coreset point instead
            for i in range(non_coreset_size):
                non_coreset_point = non_coreset[i]
                if sufficient_stat[non_coreset_point] < min_sufficient_stat:
                    min_sufficient_stat = sufficient_stat[non_coreset_point]
                    best_idx_into_non_coreset = i
            # Select whichever point reduced MMD the most
            if best_idx_into_non_coreset != -1:
                best_point = non_coreset[best_idx_into_non_coreset]
                # To ensure uniqueness, remove best_point from the non-coreset list and
                # replace with swapped out coreset_point
                non_coreset[best_idx_into_non_coreset] = coreset_point
            else:
                best_point = coreset_point
        else:
            # All points are valid candidates for swapping in
            best_point = 0
            min_sufficient_stat = sufficient_stat[0]
            for i in range(n):
                stat_i = sufficient_stat[i]
                if stat_i < min_sufficient_stat:
                    min_sufficient_stat = stat_i
                    best_point = i
        
        # Add best point to coreset and its contribution to sufficient stat
        coreset[idx_into_coreset] = best_point
        for i in range(n):
            sufficient_stat[i] += K[best_point,i]*two_over_coreset_size
            
        # Stop when all initial coreset points have been swapped
        idx_into_coreset += 1
        if idx_into_coreset == coreset_size:
            break
        # Otherwise, prepare next coreset point for swap out
        coreset_point = coreset[idx_into_coreset]
