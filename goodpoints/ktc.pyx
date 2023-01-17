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
                  double[:] aux_double_mem,
                  long[:,:] aux_long_mem,
                  long[:] output_indices) nogil:
    
    """Produces a kt.thin(m = 1) coreset of size n//2 given
    a size n x n kernel matrix K and stores it in output_indices.
    Uses order n^2 memory, as a kernel matrix is maintained in memory.
    
    Note: Assumes that n is even.
    Note: Returned coreset indices need not be sorted.
    
    Args:
      K: Matrix of KT-SPLIT kernel evaluations with shape (n, n)
      rng: random number generator
      delta: Runs halve_K with failure probability parameter delta
      unique: If True, constrains the output to never contain the same row index 
        more than once and symmetrizes output by returning the chosen coreset
        or its complement with equal probability
      aux_double_mem: scratch space array of size 2n; will be modified in place
      aux_long_mem: scratch space array of size (2, n//2); will be modified in place
      output_indices: array of size n//2, would store the output row indices into K
       representing coreset; will be modified in place
    """
    cdef long n = K.shape[0]
    
    if n == 4:
        # Use optimized implementation when n == 4
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
    
    # Compute the row means of K (i.e., meanK = K.mean(axis=1))
    # for best_K and refine_K
    # Store in first half of aux_double_mem
    cdef double[:] meanK = aux_double_mem[:n]
    cdef const double[:] K_i
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
                        long[:] coreset) nogil:
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
cpdef void halve_K(const double[:,:] K,
                   const double delta,
                   const double[:] uniforms,
                   long[:,:] coresets) nogil:    
    """Identifies two KT-SPLIT coresets of size floor(n/2) and stores them in coresets.
    This is a faster implementation of kt.split_K with m = 1.
    Uses order n^2 memory, as a kernel matrix is maintained in memory.
    
    Args:
      K: Matrix of KT-SPLIT kernel evaluations with shape (n, n)
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      uniforms: Array of floor(n/2) independent random variables uniformly distributed on 
        the unit interval [0,1]
      coresets: Array of shape (2, floor(n/2)) for storing KT-SPLIT coresets; will be
        modified in place
    """
    
    cdef long n = K.shape[0]
    cdef long num_points_in_coreset = n//2
    
    cdef double sig_sqd = 0
    cdef double sig_sqd_update
    cdef double log_multiplier = 2*log(2*n/delta)
    
    cdef long i, point0, point1
    cdef double b_sqd, thresh, alpha, prob_point1
    cdef long j, c1j, c0j
    for i in range(num_points_in_coreset):
        point0 = 2*i
        point1 = 2*i+1
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KT Best Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef const long[:] best_K(const double[:,:] K,
                          const long[:,:] coresets,
                          const double[:] meanK,
                          const long[:] best_coreset) nogil:
    """Selects the candidate coreset with smallest MMD to all input points in X (after comparing with
    a baseline standard thinning coreset).  
    Here X is implicitly represented by its kernel matrix K satisfying K[ii,:] = kernel(X[ii], X).
    
    Args:
      K: Matrix of kernel evaluations with shape (n, n)
      coresets: 2D array with each row specifying the row indices of X belonging to a coreset
      meanK: Array of length n with meanK[ii] = mean of K[ii,:]
      best_coreset: 1D array containing a coreset produced by standard thinning
    """
    cdef long num_coresets = coresets.shape[0]
    # Compute the relative MMD^2 of a standard thinning coreset
    # RelMMD^2 = MMD^2 - np.mean(K) = np.mean(K[coreset][:,coreset]) - 2*np.mean(K[coreset, :]))
    cdef double best_rel_mmd2 = squared_emp_rel_mmd_K(K, best_coreset, meanK)
    cdef double rel_mmd2
    cdef long i
    # Select the better of standard thinning coreset and the best input coreset
    if num_coresets == 2:
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
                                  const double[:] meanK) nogil:
    """Computes squared empirical relative MMD between a distribution weighted equally 
    on all points and one weighted equally on the indices in coreset.
    RelMMD^2 = MMD^2 - np.mean(K) = np.mean(K[coreset][:,coreset]) - 2*np.mean(K[coreset, :]))
    
    Args:
      K: Matrix of pairwise kernel evaluations with shape (n, n)
      coreset: Row indices of K representing coreset
      meanK: Array of length n with meanK[ii] = mean of K[ii,:]
    """
    cdef long coreset_size = coreset.shape[0]
    cdef long sqd_coreset_size = coreset_size * coreset_size
    
    # Compute rel_mmd2 = np.mean(K[coreset][:,coreset]) - 2*np.mean(meanK[coreset])
    cdef double rel_mmd2 = 0
    cdef double K_cset_i_coreset_sum
    cdef long i, j, cset_i
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
                    long[:] non_coreset) nogil:
    """
    Replaces each element of a coreset in turn by the input point that yields the greateest 
    decrease in MMD between all input points and the resulting coreset. Here X is implicitly
    represented by its kernel matrix K satisfying K[ii,:] = kernel(X[ii], X).
    Uses order n^2 memory, as a kernel matrix is maintained in memory.
    
    Note: Returned coreset and non_coreset indices need not be sorted.
    
    Args:
      K: Matrix of kernel evaluations with shape (n, n)
      coreset: Row indices of K representing coreset; will be modified in place
      meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
        used to speed up computation when not None
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
    cdef long non_coreset_size = non_coreset.shape[0]
    
    # Compute sufficient statistic representing how much each point would change the quantity
    # coreset_size * MMD^2(P,Q) if added into the coreset with weight 1/coreset_size
    #   coreset_size * MMD^2(P,Q) = coreset_size * (PPk - 2QPK + QQK)
    #   impact of adding (1/coreset_size) delta_y to Q = -2 delta_y PK + delta_y delta_y K / coreset_size + 2 delta_y Q K 
    # Statistic will later be adjusted to remove the influence of an eliminated point
    cdef long i, j
    cdef double meanK_i_coreset
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
