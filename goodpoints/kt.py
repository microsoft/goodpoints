"""Kernel Thinning.

Implementations of the (generalized) kernel thinning, kt-split, and 
kt-swap algorithms of
  Raaz Dwivedi and Lester Mackey.
  Kernel Thinning.
  https://arxiv.org/pdf/2105.05842.pdf
  
  Raaz Dwivedi and Lester Mackey.
  Generalized Kernel Thinning.
  https://arxiv.org/pdf/2110.01593.pdf
  
and the optimal four-point halving algorithm of 
  Carles Domingo-Enrich, Raaz Dwivedi, and Lester Mackey.
  Compress Then Test: Powerful Kernel Testing in Near-linear Time.
  https://arxiv.org/pdf/2301.05974.pdf
"""

import numpy as np
from numpy import newaxis
from numpy.random import default_rng
from goodpoints.tictoc import tic, toc # for timing blocks of code
from goodpoints.util import fprint # for printing while flushing buffer
from goodpoints import ktc

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kernel Thinning Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def thin(X, m, split_kernel, swap_kernel, delta=0.5, seed=None, store_K=False, meanK=None, unique=False, verbose=False):
    """Returns kernel thinning coreset of size floor(n/2^m) as row indices into X
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      m: Number of halving rounds (integer >= 0)
      split_kernel: Kernel function used by KT-SPLIT (typically a square-root kernel, krt);
        split_kernel(y,X) returns array of kernel evaluations between y and each row of X
      swap_kernel: Kernel function used by KT-SWAP (typically the target kernel, k);
        swap_kernel(y,X) returns array of kernel evaluations between y and each row of X
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      seed: Random seed to set prior to generation; if None, no seed will be set
      store_K: If False, runs O(nd) space version which does not store kernel
        matrix; if True, stores n x n kernel matrix
      meanK: None or array of length n with meanK[ii] = mean of swap_kernel(X[ii], X);
        used to speed up computation when not None
      unique: If True, constrains the output to never contain the same row index more than once
      verbose: If False do not print intermediate time taken, if True print that info when m>=7
    """
    if m == 0:
        # Zero halving rounds requested
        # Return coreset containing all indices
        return(np.arange(X.shape[0], dtype=int))

    verbose = (verbose and (m>=7))
    
    fprint('Running kt.split', verbose=verbose)
    tic()
    coresets = split(X, m, split_kernel, delta=delta, seed=seed, store_K=store_K, verbose=verbose)
    toc(print_elapsed=verbose)
    
    fprint('Running kt.swap', verbose=verbose)
    tic()
    coreset = swap(X, coresets, swap_kernel, store_K=store_K, meanK=meanK, unique=unique)
    toc(print_elapsed=verbose)
    return(coreset)
    

def thin_X(X, m, split_kernel, swap_kernel, delta=0.5, seed=None, meanK=None, unique=False, verbose=False):
    """Returns kernel thinning coreset of size floor(n/2^m) as row indices into X.
    Uses O(nd) memory as kernel matrix is not stored.
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      m: Number of halving rounds (integer >= 0)
      split_kernel: Kernel function used by KT-SPLIT (often a square-root kernel, krt);
        split_kernel(y,X) returns array of kernel evaluations between y and each row of X
      swap_kernel: Kernel function used by KT-SWAP (typically the target kernel, k);
        swap_kernel(y,X) returns array of kernel evaluations between y and each row of X
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      seed: Random seed to set prior to generation; if None, no seed will be set
      meanK: None or array of length n with meanK[ii] = mean of swap_kernel(X[ii], X);
        used to speed up computation when not None
      unique: If True, constrains the output to never contain the same row index more than once
      verbose: If False do not print intermediate time taken, if True print that info when m>=7   
    """
    if m == 0:
        # Zero halving rounds requested
        # Return coreset containing all indices
        return(np.arange(X.shape[0], dtype=int))

    verbose = (verbose and (m>=7))
    
    fprint('Running kt.split', verbose=verbose)
    tic()
    coresets = split_X(X, m, split_kernel, delta=delta, seed=seed, verbose=verbose)
    toc(print_elapsed=verbose)
    
    fprint('Running kt.swap', verbose=verbose)
    tic()
    coreset = swap_X(X, coresets, swap_kernel, meanK=meanK, unique=unique)
    toc(print_elapsed=verbose)
    return(coreset)

def thin_K(K_split, K_swap, m, delta=0.5, seed=None, unique=False, verbose=False):
    """Returns kernel thinning coreset of size floor(n/2^m) as row indices into K_split.
    Uses order n^2 memory, as a kernel matrix is maintained in memory.
   
    Args:
      K_split: Kernel matrix for KT-SPLIT with shape (n, n) 
        (often a square-root kernel krt or the target kernel k) 
      K_swap: Kernel matrix used by KT-SWAP with shape (n, n) (typically the target kernel, k) 
      m: Number of halving rounds (integer >= 0)
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      seed: Random seed to set prior to generation; if None, no seed will be set
      unique: If True, constrains the output to never contain the same row index more than once
      verbose: If False do not print intermediate time taken, if True print that info when m>=7    
    """
    if m == 0:
        # Zero halving rounds requested
        # Return coreset containing all indices
        return(np.arange(K_split.shape[0], dtype=int))

    coresets = split_K(K_split, m, delta=delta, seed=seed, verbose=verbose)
    coreset = swap_K(K_swap, coresets, unique=unique)
    return(coreset)

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kernel Halving Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def opt_halve4_K(K, seed=None):
    """Identifies a two-point coreset with smallest MMD to the four-point input point set X.
    If uniform < .5, returns that coreset in coreset; otherwise, returns the complement of that 
    coreset. Here X is implicitly represented by its kernel matrix K satisfying 
    K[ii,:] = kernel(X[ii], X).
    
    Args:
      K: Matrix of pairwise kernel evaluations with shape (4, 4)
      uniform: Random real number distributed uniformly on the unit interval [0,1]
      coreset: Preallocated 1D array with length 2 representing the coreset of row indices into K
        to be returned; will be modified in place
    """
    # Allocate memory for storing coreset
    coreset = np.empty(2, dtype=int)
    # Generate uniform random number for selecting unbiased coreset
    uniform = default_rng(seed).random()
    # Run optimal halving and store coreset in place
    ktc.opt_halve4_K(K, uniform, coreset)
    return(coreset)
    

def halve_K(K, delta=0.5, seed=None):    
    """Returns two KT-SPLIT coresets of size floor(n/2) as a 2D array.
    This is a faster implementation of kt.split_K with m = 1.
    Uses order n^2 memory, as a kernel matrix is maintained in memory.
    
    Args:
      K: Matrix of KT-SPLIT kernel evaluations with shape (n, n)
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      seed: Random seed to set prior to generation; if None,
        no seed will be set
    """
    
    n = K.shape[0]
    num_points_in_coreset = n//2
    
    # Pre-generate uniform random numbers needed for kernel halving
    rng = default_rng(seed)
    uniforms = rng.random(num_points_in_coreset)
    
    # Allocate memory for storing coresets
    coresets = np.empty((2,num_points_in_coreset), dtype=int)

    # Run kernel halving and store coresets in place
    ktc.halve_K(K, delta, uniforms, coresets)
    
    return(coresets)

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KT Split Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def split(X, m, kernel, delta=0.5, seed=None, store_K=False, verbose=False):
    """Returns 2^m kernel thinning coresets of size floor(n/2^m) as a 2D array
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      m: Number of halving rounds
      kernel: Kernel function (typically a square-root kernel, krt);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      seed: Random seed to set prior to generation; if None, no seed will be set
      store_K: If False, runs O(nd) space version which does not store kernel
        matrix; if True, stores n x n kernel matrix
      verbose: If False, do not print intermediate time taken in thinning rounds, if True print that info
    """
    if store_K:
       # Store matrix of kernel evaluations between each pair of points
      return split_K(kernel_matrix(X, kernel), m, delta=delta, seed=seed, verbose=verbose)

    return(split_X(X, m, kernel, delta=delta, seed=seed, verbose=verbose))

# Constant used by split functions
TWO_LOG_2 = 2*np.log(2)

def split_X(X, m, kernel, delta=0.5, seed=None, verbose=False):
    """Returns 2^m KT-SPLIT coresets of size floor(n/2^m) as a 2D array.
    Uses O(nd) memory as kernel matrix is not stored.
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      m: Number of halving rounds
      kernel: Kernel function (typically a square-root kernel, krt);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      seed: Random seed to set prior to generation; if None, no seed will be set
      verbose: If False do not print intermediate time taken, if True print that info
    """
    if m == 0:
        # Zero halving rounds requested
        # Return 2D coreset array containing a single coreset (one row) with all indices
        return(np.arange(X.shape[0], dtype=int)[newaxis,:])
    
    verbose = verbose and (m>=7)
    # Function which returns kernel value for two arrays of row indices of X
    def k(ii, jj):
        return(kernel(X[ii], X[jj]))
    
    # Initialize random number generator
    rng = default_rng(seed)
    n, _ = X.shape
    
    # Initialize coresets, each a vector of integers indexing the rows of X
    coresets = dict()
    # Store sum of kernel evaluations between each point eventually added to a coreset
    # and all of the points previously added to that coreset
    KC = dict()

    # Initialize subGaussian parameters
    # sig_sqd[j][j2] determines the threshold for halving coresets[j][j2]
    sig_sqd = dict()
    # Store multiplier to avoid recomputing
    log_multiplier = 2*np.log(2*n*m/delta)

    for j in range(m+1):
        # Initialize coresets[j][j2] for each j2 < 2^j to array of size n/2^j 
        # with invalid -1 values
        num_coresets = int(2**j)
        
        num_points_in_coreset = n//num_coresets
        coresets[j] = np.full((num_coresets, num_points_in_coreset), -1, dtype=int)
        # Initialize associated coreset kernel sums arbitrarily
        KC[j] = np.empty((num_coresets, num_points_in_coreset))

        # Initialize subGaussian parameters to 0 
        sig_sqd[j] = np.zeros(num_coresets)
                              
    # Store kernel(xi, xi) for each point i
    diagK = np.empty(n)
        
    # If verbose---Track the time taken if m is large
    # Output timing when sample size doubles til n/2, and then every n/8 sample points
    nidx = 1
    tic()
    for i in range(n):
        # Track progress
        if i==nidx:
            fprint(f"Tracking update: Finished processing sample number {nidx}/{n}", verbose=verbose)
            toc(print_elapsed=verbose)
            tic()
            
            if nidx<int(n/2):
                nidx *= 2
            else:
                nidx += int(n/2**3)

        # Add each datapoint to coreset[0][0]
        coreset = coresets[0][0]
        coreset[i] = i
        # Capture index i as 1D array to ensure X[i_array] is a 2D array
        i_array = coreset[i, newaxis]
        # Store kernel evaluation with all points <= i
        ki = k(i_array, coreset[:(i+1)]) 
        # Store summed kernel inner product with all points < i
        KC[0][0,i] = np.sum(ki[:i]) 
        # Store diagonal element, kernel(xi, xi)
        diagK[i] = ki[i] 

        # If 2^(j+1) divides (i+1), add a point from coreset[j][j2] to each of
        # coreset[j+1][2*j2] and coreset[j+1][2*j2+1]
        for j in range(min(m, largest_power_of_two(i+1))):
            parent_coresets = coresets[j]
            child_coresets = coresets[j+1]
            parent_KC = KC[j]
            child_KC = KC[j+1]
            num_parent_coresets = parent_coresets.shape[0]
            # j_log_multiplier = 2*np.log(2*n*m/delta/2^j) 
            #                  = 2*np.log(2*n*m/delta) - j * 2 log(2)
            #                  = log_multiplier - j * TWO_LOG_2
            # the term is 2^{j-1} in the paper because j starts at 1; here j starts at 0
            j_log_multiplier = log_multiplier - j * TWO_LOG_2
            # Consider each parent coreset in turn
            for j2 in range(num_parent_coresets):
                parent_coreset = parent_coresets[j2]
                #tic()
                # Number of points in parent_coreset
                parent_idx = (i+1) // num_parent_coresets
                # Get last two points from the parent coreset
                # newaxis ensures array dimensions are appropriate for kernel function
                point1, point2 = parent_coreset[parent_idx-2, newaxis], parent_coreset[parent_idx-1, newaxis]
                # Compute kernel(x1, x2)
                K12 = k(point1,point2)

                # Use adaptive failure threshold
                # Compute b^2 = ||f||^2 = ||k(x1,.) - k(x2,.)||_k^2
                b_sqd = diagK[point2] + diagK[point1] - 2*K12
                # Update threshold for halving parent coreset
                # a = max(b sig sqrt(j_log_multiplier), b^2)
                thresh = max(np.sqrt(sig_sqd[j][j2]*b_sqd*j_log_multiplier), b_sqd)
                if sig_sqd[j][j2] == 0:
                    sig_sqd[j][j2] = b_sqd
                elif thresh != 0:
                    # Note: If threshold is zero, b_sqd is zero so sigma does not change
                    # If thresh != 0, update subGaussian parameter
                    # s^2 += 2*b^2*(.5 + (b^2/(2 a) - 1)*s^2/a)_+
                    sig_sqd_update = .5 + (b_sqd/(2*thresh) - 1)*sig_sqd[j][j2]/thresh
                    if sig_sqd_update > 0:
                        sig_sqd[j][j2] += 2*b_sqd*sig_sqd_update
                # To avoid division by zero, set zero threshold to arbitrary positive value
                # (Does not impact algorithm correctness as b_sqd = 0 as well)
                if thresh == 0: thresh = 1.

                # Compute inner product with other points in parent coreset:
                #  sum_{l < parent_idx-2} <k(coreset[j][l],.), k(x1, .) - k(x2, .)>
                # Note that KC[j, point1] = <k(coreset[j][l],.), k(x1, .)> and
                # KC[j, point2] = <k(coreset[j][l],.), k(x2, .)> + k(x1,x2)         
                if parent_idx > 2:
                    alpha = parent_KC[j2, parent_idx-2] - parent_KC[j2, parent_idx-1] + K12 
                else:
                    alpha = 0.
                # Identify the two new child coresets
                left_child_coreset = child_coresets[2*j2]
                right_child_coreset = child_coresets[2*j2+1]
                # Number of points in each new coreset
                child_idx = (parent_idx//2)-1
                if child_idx > 0:
                    # Subtract 2 * inner product with all points in left child coreset:
                    # - 2 * sum_{l < child_idx} <k(coreset[j][l],.), k(x1, .) - k(x2, .)> 
                    child_points = left_child_coreset[:child_idx]
                    point1_kernel_sum = np.sum(k(point1,child_points))
                    point2_kernel_sum = np.sum(k(point2,child_points))
                    alpha -= 2*(point1_kernel_sum - point2_kernel_sum)
                else:
                    point1_kernel_sum = 0
                    point2_kernel_sum = 0
                # Add point2 to coreset[j] with probability prob_point2; add point1 otherwise
                prob_point2 = 0.5*(1-alpha/thresh)
                if rng.random() <= prob_point2:
                    left_child_coreset[child_idx] = point2
                    right_child_coreset[child_idx] = point1
                    child_KC[2*j2, child_idx] = point2_kernel_sum
                    child_KC[2*j2+1, child_idx] = point1_kernel_sum
                else:
                    left_child_coreset[child_idx] = point1
                    right_child_coreset[child_idx] = point2
                    child_KC[2*j2, child_idx] = point1_kernel_sum
                    child_KC[2*j2+1, child_idx] = point2_kernel_sum

    # Return coresets of size floor(n/2^m)
    return(coresets[m])

def split_K(K, m, diagK=None, delta=0.5, seed=None, verbose=False):
    """Returns 2^m KT-SPLIT coresets of size floor(n/2^m) as a 2D array.
    Uses order n^2 memory, as a kernel matrix is maintained in memory.
    
    Args:
      K: Matrix of KT-SPLIT kernel evaluations with shape (n, n)
      m: Number of halving rounds
      diagK: None or array of length n with diagK[ii] = K[ii,ii];
        used to speed up computation when not None
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      seed: Random seed to set prior to generation; if None,
        no seed will be set
      verbose: If False do not print intermediate time taken, if True print that info
    """
    if m == 0:
        # Zero halving rounds requested
        # Return 2D coreset array containing a single coreset (one row) with all indices
        return(np.arange(K.shape[0], dtype=int)[newaxis,:])
    
    if m == 1:
        # Use streamlined kernel halving implementation
        return(halve_K(K, delta=delta, seed=seed))
    
    # Initialize random number generator
    rng = default_rng(seed)
    n, _ = K.shape
    
    # Initialize coresets, each a vector of integers indexing the rows of X
    coresets = dict()
    # Store sum of kernel evaluations between each point eventually added to a coreset
    # and all of the points previously added to that coreset
    KC = dict()

    # Initialize subGaussian parameters
    # sig_sqd[j][j2] determines the threshold for halving coresets[j][j2]
    sig_sqd = dict()
    # Store multiplier to avoid recomputing
    log_multiplier = 2*np.log(2*n*m/delta)
        
    coresets[0] = np.arange(K.shape[0], dtype=int)[newaxis,:]
    KC[0] = np.empty((1, n))
    sig_sqd[0] = np.zeros(1)
    
    for j in range(1,m+1):
        # Initialize coresets[j][j2] for each j2 < 2^j to array of size n/2^j 
        # with invalid -1 values
        num_coresets = int(2**j)
        
        num_points_in_coreset = n//num_coresets
        coresets[j] = np.empty((num_coresets, num_points_in_coreset), dtype=int)
        # Initialize associated coreset kernel sums arbitrarily
        KC[j] = np.empty((num_coresets, num_points_in_coreset))

        # Initialize subGaussian parameters to 0 
        sig_sqd[j] = np.zeros(num_coresets)
                              
    # Store kernel(xi, xi) for each point i
    if diagK is None: diagK = np.diag(K)
    
    # If verbose---Track the time taken
    # Output timing when sample size doubles til n/2, and then every n/8 sample points
    nidx = 1
    tic()
    KC[0][0,:] = np.sum(np.tril(K,k=-1),axis=1)
    for i in range(n):
        # Track progress
        if i==nidx:
            fprint(f"Tracking update: Finished processing sample number {nidx}/{n}", verbose=verbose)
            toc(print_elapsed=verbose)
            tic()
            if nidx<int(n/2):
                nidx *= 2
            else:
                nidx += int(n/2**3)

        # If 2^(j+1) divides (i+1), add a point from coreset[j][j2] to each of
        # coreset[j+1][2*j2] and coreset[j+1][2*j2+1]
        for j in range(i % 2) if m == 1 else range(min(m, largest_power_of_two(i+1))):
            parent_coresets = coresets[j]
            child_coresets = coresets[j+1]
            parent_KC = KC[j]
            child_KC = KC[j+1]
            num_parent_coresets = parent_coresets.shape[0]
            # j_log_multiplier = 2*np.log(2*n*m/delta/2^j) 
            j_log_multiplier = log_multiplier - j * TWO_LOG_2
            # Consider each parent coreset in turn
            for j2 in range(num_parent_coresets):
                parent_coreset = parent_coresets[j2]
                # Number of points in parent_coreset
                parent_idx = (i+1) // num_parent_coresets
                # Get last two points from the parent coreset
                point1, point2 = parent_coreset[parent_idx-2], parent_coreset[parent_idx-1]
                # Compute kernel(x1, x2)
                K12 = K[point1,point2]

                # Use adaptive failure threshold
                # Compute b^2 = ||f||^2 = ||k(x1,.) - k(x2,.)||_k^2
                b_sqd = diagK[point2] + diagK[point1] - 2*K12
                # Update threshold for halving parent coreset
                # a = max(b sig sqrt(j_log_multiplier), b^2)
                thresh = max(np.sqrt(sig_sqd[j][j2]*b_sqd*j_log_multiplier), b_sqd)
                if sig_sqd[j][j2] == 0:
                    sig_sqd[j][j2] = b_sqd
                elif thresh != 0:
                    # Note: If threshold is zero, b_sqd is zero so sigma does not change
                    # If thresh != 0, update subGaussian parameter
                    # s^2 += 2*b^2*(.5 + (b^2/(2 a) - 1)*s^2/a)_+
                    sig_sqd_update = .5 + (b_sqd/(2*thresh) - 1)*sig_sqd[j][j2]/thresh
                    if sig_sqd_update > 0:
                        sig_sqd[j][j2] += 2*b_sqd*sig_sqd_update
                # To avoid division by zero, set zero threshold to arbitrary positive value
                # (Does not impact algorithm correctness as b_sqd = 0 as well)
                if thresh == 0: thresh = 1.
                        
                # Compute inner product with other points in parent coreset:
                #  sum_{l < parent_idx-2} <k(coreset[j][l],.), k(x1, .) - k(x2, .)>
                # Note that KC[j, point1] = <k(coreset[j][l],.), k(x1, .)> and
                # KC[j, point2] = <k(coreset[j][l],.), k(x2, .)> + k(x1,x2)         
                if parent_idx > 2:
                    alpha = parent_KC[j2, parent_idx-2] - parent_KC[j2, parent_idx-1] + K12 
                else:
                    alpha = 0.
                # Identify the two new child coresets
                left_child_coreset = child_coresets[2*j2]
                right_child_coreset = child_coresets[2*j2+1]
                # Number of points in each new coreset
                child_idx = (parent_idx//2)-1
                if child_idx > 0:
                    # Subtract 2 * inner product with all points in left child coreset:
                    # - 2 * sum_{l < child_idx} <k(coreset[j+1][l],.), k(x1, .) - k(x2, .)> 
                    child_points = left_child_coreset[:child_idx]
                    point1_kernel_sum = np.sum(K[point1,child_points])
                    point2_kernel_sum = np.sum(K[point2,child_points]) 
                    alpha -= 2*(point1_kernel_sum - point2_kernel_sum)
                else:
                    point1_kernel_sum = 0
                    point2_kernel_sum = 0
                # Add point2 to coreset[j+1] with probability prob_poin22; add point1 otherwise
                prob_point2 = 0.5*(1-alpha/thresh)
                if rng.random() <= prob_point2:
                    left_child_coreset[child_idx] = point2
                    right_child_coreset[child_idx] = point1
                    child_KC[2*j2, child_idx] = point2_kernel_sum
                    child_KC[2*j2+1, child_idx] = point1_kernel_sum
                else:
                    left_child_coreset[child_idx] = point1
                    right_child_coreset[child_idx] = point2
                    child_KC[2*j2, child_idx] = point1_kernel_sum
                    child_KC[2*j2+1, child_idx] = point2_kernel_sum

    # Return coresets of size floor(n/2^m)
    return(coresets[m])

def kernel_matrix(X, kernel):
    """Returns the kernel matrix of shape (n, n) with rows 
    K[ii] = kernel(X[ii, newaxis], X)
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      kernel: Kernel function;
        kernel(y,X) returns array of kernel evaluations between y and 
        each row of X
    """
    n = X.shape[0]
    K = np.empty((n,n))
    for ii in range(n):
        K[ii] = kernel(X[ii, newaxis], X)
    return K

def largest_power_of_two(n):
    """Returns the largest j such that 2**j divides n
       Based on 
         https://www.geeksforgeeks.org/highest-power-of-two-that-divides-a-given-number/#:~:text=Highest%20power%20of%202%20that%20divides%205%20is%201.
         https://stackoverflow.com/questions/13105875/compute-fast-log-base-2-ceiling-in-python
    """
    return((n & (~(n - 1))).bit_length() - 1)


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KT Swap Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def swap(X, coresets, kernel, store_K=False, meanK=None, unique=False):
    """Selects the candidate coreset with smallest MMD to all points in X (after comparing with
    a baseline standard thinning coreset) and iteratively refine that coreset.
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      coresets: 2D array with each row specifying the row indices of X belonging to a coreset
      kernel: Kernel function (typically the target kernel, k);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      store_K: If False, runs O(nd) space version which does not store kernel
        matrix; if True, stores n x n kernel matrix
      meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
        used to speed up computation when not None
      unique: If True, constrains the output to never contain the same row index more than once
    """
    if store_K:
      # Store matrix of kernel evaluations between each pair of points
      return swap_K(kernel_matrix(X, kernel), coresets, meanK=meanK, unique=unique)
    
    return swap_X(X, coresets, kernel, meanK=meanK, unique=unique)

def swap_X(X, coresets, kernel, meanK=None, unique=False):
    """Selects the candidate coreset with smallest MMD to all points in X (after comparing with
    a baseline standard thinning coreset) and iteratively refine that coreset.
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      coresets: 2D array with each row specifying the row indices of X belonging to a coreset
      kernel: Kernel function (typically the target kernel, k);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
        used to speed up computation when not None
      unique: If True, constrains the output to never contain the same row index more than once
    """
    # Compute meanK if appropriate
    if meanK is None: meanK = kernel_matrix_row_mean(X, kernel)
    # Return refined version of best coreset
    return(refine_X(X, best_X(X, coresets, kernel, meanK=meanK), kernel, meanK=meanK, unique=unique))

def swap_K(K, coresets, meanK=None, unique=False):
    """Selects the candidate coreset with smallest MMD to all input points in X (after comparing with
    a baseline standard thinning coreset) and iteratively refine that coreset.
    Here X is implicitly represented by its kernel matrix K satisfying K[ii,:] = kernel(X[ii], X).
    
    Args:
      K: Matrix of kernel evaluations with shape (n, n)
      coresets: 2D array with each row specifying the row indices of X belonging to a coreset
      meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
        used to speed up computation when not None
      unique: If True, constrains the output to never contain the same row index more than once
    """
    if meanK is None: meanK = K.mean(axis=1)
    coreset = best_K(K, coresets, meanK=meanK)
    return(refine_K(K, coreset, meanK=meanK, unique=unique))
 
def kernel_matrix_row_mean(X, kernel):
    """Returns the mean of each kernel matrix row
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      kernel: Kernel function; 
          kernel(y,X) returns array of kernel evaluations between y and each row of X
          
    Returns array of length n with ii-th entry equal to the mean of kernel(X[ii], X)
    """
    n = X.shape[0]
    meanK = np.empty(n)
    for ii in range(n):
        # Mean kernel evaluation between xi and every row in X
        meanK[ii] =  np.mean(kernel(X[ii, newaxis], X))
    return(meanK)    

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KT Best Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def best_X(X, coresets, kernel, meanK=None):
    """
    Selects the candidate coreset with smallest MMD to all points in X (after comparing with
    a baseline standard thinning coreset).
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      coresets: 2D array with each row specifying the row indices of X belonging to a coreset
      kernel: kernel(x, X) (typically the target kernel, k);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
        used to speed up computation when not None
    """
    n = X.shape[0]
            
    # Compute the relative MMD^2 of a standard thinning coreset
    # RelMMD^2 = MMD^2 - np.mean(K) = np.mean(K[coreset][:,coreset]) - 2*np.mean(K[coreset, :]))
    coreset_size = coresets.shape[1]
    # Initialize as a standard thinned coreset from the end
    best_coreset = np.array(range(n-1,-1,-(n//coreset_size)))[::-1] 
    best_rel_mmd2 = squared_emp_rel_mmd_X(X, best_coreset, kernel, meanK=meanK)
    
    # Select the better of standard thinning coreset and the best input coreset
    for coreset in coresets:
        rel_mmd2 = squared_emp_rel_mmd_X(X, coreset, kernel, meanK=meanK)
        if rel_mmd2 < best_rel_mmd2:
            best_rel_mmd2 = rel_mmd2
            best_coreset = coreset
    
    return(best_coreset)

def best_K(K, coresets, meanK=None):
    """
    Selects the candidate coreset with smallest MMD to all input points in X (after comparing with
    a baseline standard thinning coreset).  
    Here X is implicitly represented by its kernel matrix K satisfying K[ii,:] = kernel(X[ii], X).
    
    Args:
      K: Matrix of kernel evaluations with shape (n, n)
      coresets: 2D array with each row specifying the row indices of X belonging to a coreset
      meanK: None or array of length n with meanK[ii] = mean of K[ii,:];
        used to speed up computation when not None
    """
    n = K.shape[0]
    coreset_size = coresets.shape[1]
    if meanK is None: meanK = K.mean(axis=1)
            
    # Consider a standard thinned coreset from the end as initial best_coreset
    best_coreset = np.flip(np.arange(n-1,-1,-(n//coreset_size)))
    # Select better of best_coreset and the best coreset in coresets
    return(np.asarray(ktc.best_K(K, coresets, meanK, best_coreset)))

def squared_emp_rel_mmd_X(X, coreset, kernel, meanK=None): 
    """Computes squared empirical relative MMD between a distribution weighted equally 
    on all points and one weighted equally on the indices in coreset.
    RelMMD^2 = MMD^2 - np.mean(K) = np.mean(K[coreset][:,coreset]) - 2*np.mean(K[coreset, :]))
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      coreset: Row indices of X representing coreset
      kernel: Kernel function (typically the target kernel, k);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      meanK: array of length n with meanK[ii] = mean of kernel(X[ii], X)
    """
    # Note: K[coreset][:,coreset] returns the appropriate coreset x coreset submatrix 
    # while K[coreset, coreset] does not (it returns [K[i,i] for i in coreset])
    # Keep track of sum_{ii in coreset} mean(K[ii][:,coreset])
    k_core_core = 0.
    # Keep track of sum_{ii in coreset} mean(K[ii,:])
    k_core_all = 0.
    if meanK is None:
        for ii in coreset:
            # Kernel evaluations between x_{ii} and all points in X
            kii = kernel(X[ii, newaxis], X)
            k_core_core += np.mean(kii[coreset])
            k_core_all += np.mean(kii)
    else:
        k_core_all = np.sum(meanK[coreset])
        for ii in coreset:
            # Add mean kernel evaluation between x_{ii} and all coreset points in X
            k_core_core += np.mean(kernel(X[ii, newaxis], X[coreset]))
        
    # Return mean(K[coreset][:,coreset]) - 2 * mean(K[coreset,:])
    coreset_size = len(coreset)
    return((k_core_core - 2*k_core_all)/coreset_size)
    
'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KT Refine Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def refine_X(X, coreset, kernel, meanK=None, unique=False):
    """
    Replaces each element of a coreset in turn by the point in X that yields the minimum 
    MMD between all points in X and the resulting coreset.
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      coreset: Row indices of X representing coreset
      kernel: Kernel function (typically the target kernel, k);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
        used to speed up computation when not None
      unique: If True, constrains the output to never contain the same row index more than once
              (logic of point-by-point swapping is altered to ensure MMD improvement
              as well as that the coreset does not contain any repeated points at any iteration)
    """
    n = X.shape[0]

    # Initialize new KT coreset to original coreset
    coreset = np.copy(coreset)
    coreset_size = len(coreset)
    
    # Initialize sufficient kernel matrix statistics
    # sufficient_stat = twoncoresumK + ndiagK - twomeanK where
    #   ndiagK[ii] = diagonal element of kernel matrix, kernel(X[ii], X[ii]) / coreset_size
    #   twomeanK[ii] = 2 * the mean of kernel(X[ii], X)
    #   twoncoresumK[ii] = 2 * the sum of kernel(X[ii], X[coreset]) / coreset_size
    two_over_coreset_size = 2/coreset_size
    
    # Initialize coreset indicator of size n, takes value True at coreset indices
    coreset_indicator = np.zeros(n, dtype=bool)
    coreset_indicator[coreset] = True
    
    if meanK is None:
        sufficient_stat = np.empty(n)
        for ii in range(n):
            # if unique then set sufficient_stat for coreset indices to infinity
            if unique and coreset_indicator[ii]:
                sufficient_stat[ii] = np.inf
            else:
                # Kernel evaluation between xi and every row in X
                kii =  kernel(X[ii, newaxis], X)
                sufficient_stat[ii] = 2*(np.mean(kii[coreset])-np.mean(kii)) + kii[ii]/coreset_size               
    else:
        # Initialize to kernel diagonal normalized by coreset_size - 2 * meanK
        sufficient_stat = kernel(X, X)/coreset_size - 2 * meanK
        # Add in contribution of coreset
        for ii in range(n):
            # if unique then set sufficient_stat for coreset indices to infinity
            if unique and coreset_indicator[ii]:
                sufficient_stat[ii] = np.inf
            else:
                # Kernel evaluation between xi and every coreset row in X
                kiicore =  kernel(X[ii, newaxis], X[coreset])
                sufficient_stat[ii] += 2*np.mean(kiicore)
    
    # Consider each coreset point in turn 
    for coreset_idx in range(coreset_size):
        # if unique have to compute sufficient stats for the current coreset point
        if unique:
            # initially all coreset indices have sufficient_stat set to infinity; 
            # before altering any coreset point, we compute the sufficient_stat for it, and
            # compare replacing it with every other *non coreset* point; the best point (bp)
            # then takes the spot of the point in consideration with bp's sufficient_stat set to infty.
            # thus at each iteration of the for loop, all the current coreset elements except the one 
            # in consideration have sufficient_stats set to infty
            cidx = coreset[coreset_idx] # the index of coreset_idx in X
            if meanK is None:
                # Kernel evaluation between x at cidx and every row in X
                kcidx = kernel(X[cidx, newaxis], X)
                sufficient_stat[cidx] = 2*(np.mean(kcidx[coreset])-np.mean(kcidx)) + kcidx[cidx]/coreset_size    
            else:
                kcidxcore =  kernel(X[cidx, newaxis], X[coreset])
                sufficient_stat[cidx] = kernel(X[cidx, newaxis], X[cidx, newaxis])/coreset_size - 2 * meanK[cidx] 
                sufficient_stat[cidx] += 2*np.mean(kcidxcore)
            
        # Remove the contribution of coreset_idx point from the normalized coreset sum in sufficient stat
        sufficient_stat -= kernel(X[coreset[coreset_idx], newaxis], X)*two_over_coreset_size
        # Find the input point that would reduce MMD the most
        best_point = np.argmin(sufficient_stat)
        # Add best point to coreset and its contribution to sufficient stat
        coreset[coreset_idx] = best_point
        sufficient_stat += kernel(X[best_point, newaxis], X)*two_over_coreset_size
        if unique:
            sufficient_stat[best_point] = np.inf
    return(coreset)
    
def refine_K(K, coreset, meanK=None, unique=False):
    """
    Replaces each element of a coreset in turn by the point in X that yields the minimum 
    MMD between all points in X and the resulting coreset. Here X is implicitly
    represented by its kernel matrix K satisfying K[ii,:] = kernel(X[ii], X).
    
    Args:
      K: Matrix of kernel evaluations with shape (n, n)
      coreset: Row indices of K representing coreset
      meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
        used to speed up computation when not None
      unique: If True, constrains the output to never contain the same row index more than once
    """
    
    n = K.shape[0]
    if meanK is None: meanK = K.mean(axis=1)

    # Make a copy of the coreset to be modified in place
    coreset = np.copy(coreset)
    # Allocate memory for array of sufficient statistics, one per input point
    sufficient_stat = np.empty(n)
    if unique:
        # Identify the indices of non-coreset points
        non_coreset = np.ones(n, dtype=bool)
        non_coreset[coreset] = False
        non_coreset = np.where(non_coreset)[0]
    else:
        # If unique is False, non_coreset argument is not used
        non_coreset = None
    # Run refine and update coresets in place
    ktc.refine_K(K, coreset, meanK, sufficient_stat, unique, non_coreset)
    
    return(coreset)
    



