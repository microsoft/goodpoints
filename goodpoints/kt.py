"""Kernel Thinning.

Implementations of the (generalized) kernel thinning, kt-split, and 
kt-swap algorithms of
  Raaz Dwivedi and Lester Mackey.
  Kernel Thinning.
  https://arxiv.org/pdf/2105.05842.pdf
  
  Raaz Dwivedi and Lester Mackey.
  Generalized Kernel Thinning.
  https://arxiv.org/pdf/2110.01593.pdf
"""

import numpy as np
import numpy.random as npr
from goodpoints.tictoc import tic, toc # for timing blocks of code
from goodpoints.util import fprint # for printing while flushing buffer

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Kernel Thinning Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def thin(X, m, split_kernel, swap_kernel, delta=0.5, seed=None, store_K=False, meanK=None, unique=False):
    """Returns kernel thinning coreset of size floor(n/2^m) as row indices into X
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      m: Number of halving rounds
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
    """
    # Partition points into 2^m candidate coresets of size floor(n/2^m)
    coresets = split(X, m, split_kernel, delta=delta, seed=seed, store_K=store_K)
    # Select best coreset and iteratively refine
    return(swap(X, coresets, swap_kernel, store_K=store_K, meanK=meanK,unique=unique))

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KT Split Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

def split(X, m, kernel, delta=0.5, seed=None, store_K=False):
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
    """
    return(split_K(X, m, kernel, delta=delta, seed=seed) if store_K
           else split_X(X, m, kernel, delta=delta, seed=seed))

def split_X(X, m, kernel, delta=0.5, seed=None):
    """Returns 2^m kernel thinning coresets of size floor(n/2^m) as a 2D array
    (uses O(nd) space, memory efficient for small d; slower in computation time
    compared to split_K,
    should be preferred for large n when n * n memory becomes an issue)
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      m: Number of halving rounds
      kernel: Kernel function (typically a square-root kernel, krt);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      seed: Random seed to set prior to generation; if None, no seed will be set
    """
    # Function which returns kernel value for two arrays of row indices of X
    def k(ii, jj):
        return(kernel(X[ii], X[jj]))
    
    # Initialize random number generator
    rng = npr.default_rng(seed)
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
    log_multiplier = 2*np.log(4*n/delta)

    for j in range(m+1):
        # Initialize coresets[j][j2] for each j2 < 2^j to array of size n/2^j 
        # with invalid -1 values
        num_coresets = int(2**j)
        
        num_points_in_coreset = int(n/num_coresets)
        coresets[j] = np.full((num_coresets, num_points_in_coreset), -1, dtype=int)
        # Initialize associated coreset kernel sums arbitrarily
        KC[j] = np.empty((num_coresets, num_points_in_coreset))

        # Initialize subGaussian parameters to 0 
        sig_sqd[j] = np.zeros(num_coresets)
                              
    # Store kernel(xi, xi) for each point i
    diagK = np.empty(n)
        
    # Track progress when m is large
    # Output timing when sample size doubles til n/2, and then every n/8 sample points
    sample_wise_tictoc = False
    nidx = 1
    if m >= 6:
        sample_wise_tictoc = True
        tic()
    for i in range(n):
        # Track progress
        if (i==nidx) and sample_wise_tictoc:
            fprint(f"special tracking update: Finished processing sample number {nidx}/{n}")
            toc()
            tic()
            if nidx<int(n/2):
                nidx *= 2
            else:
                nidx += int(n/2**3)

        # Add each datapoint to coreset[0][0]
        coreset = coresets[0][0]
        coreset[i] = i
        # Capture index i as 1D array to ensure X[i_array] is a 2D array
        i_array = coreset[i, np.newaxis]
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
            # Consider each parent coreset in turn
            for j2 in range(num_parent_coresets):
                parent_coreset = parent_coresets[j2]
                #tic()
                # Number of points in parent_coreset
                parent_idx = int((i+1) / num_parent_coresets)
                # Get last two points from the parent coreset
                # np.newaxis ensures array dimensions are appropriate for kernel function
                point1, point2 = parent_coreset[parent_idx-1, np.newaxis], parent_coreset[parent_idx-2, np.newaxis]
                # Compute kernel(x1, x2)
                K12 = k(point1,point2)

                # Use adaptive failure threshold
                # Compute b^2 = ||f||^2 = ||k(x1,.) - k(x2,.)||_k^2
                b_sqd = diagK[point2] + diagK[point1] - 2*K12
                # Update threshold for halving parent coreset
                # a = max(b sig sqrt(log_multiplier), b^2)
                thresh = max(np.sqrt(sig_sqd[j][j2]*b_sqd*log_multiplier), b_sqd)
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
                # Note that KC[j, point2] = <k(coreset[j][l],.), k(x2, .)> and
                # KC[j, point1] = <k(coreset[j][l],.), k(x1, .)> + k(x1,x2)         
                if parent_idx > 2:
                    alpha = parent_KC[j2, parent_idx-1] - parent_KC[j2, parent_idx-2] - K12 
                else:
                    alpha = 0.
                # Identify the two new child coresets
                left_child_coreset = child_coresets[2*j2]
                right_child_coreset = child_coresets[2*j2+1]
                # Number of points in each new coreset
                child_idx = int(parent_idx/2-1)
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
                # Add point2 to coreset[j] with probability prob_poin22; add point1 otherwise
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

def split_K(X, m, kernel, c=None, delta=0.5, seed=None):
    """Returns 2^m kernel thinning coresets of size floor(n/2^m) as a 2D array
    (uses O(n^2) space, faster in computation time compared to split_X,
    should be preferred for small n or when d > n)
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      m: Number of halving rounds
      kernel: Kernel function (typically a square-root kernel, krt);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      delta: Run KT-SPLIT with constant failure probabilities delta_i = delta/n
      seed: Random seed to set prior to generation; if None,
        no seed will be set
    """
    # Function which returns kernel value for two arrays of row indices of X
    def k(ii, jj):
        return(kernel(X[ii], X[jj]))
    
    # Initialize random number generator
    rng = npr.default_rng(seed)
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
    log_multiplier = 2*np.log(4*n/delta)
        
    for j in range(m+1):
        # Initialize coresets[j][j2] for each j2 < 2^j to array of size n/2^j 
        # with invalid -1 values
        num_coresets = int(2**j)
        
        num_points_in_coreset = int(n/num_coresets)
        coresets[j] = np.full((num_coresets, num_points_in_coreset), -1, dtype=int)
        # Initialize associated coreset kernel sums arbitrarily
        KC[j] = np.empty((num_coresets, num_points_in_coreset))

        # Initialize subGaussian parameters to 0 
        sig_sqd[j] = np.zeros(num_coresets)
                              
    # Store kernel evaluations between each point i and all points <= i
    K = np.empty((n,n))
    # Store kernel(xi, xi) for each point i
    diagK = np.empty(n)
    
    # Track progress when m is large
    # Output timing when sample size doubles til n/2, and then every n/8 sample points
    sample_wise_tictoc = False
    nidx = 1
    if m >= 6:
        sample_wise_tictoc = True
        tic()
    for i in range(n):
        # Track progress
        if (i==nidx) and sample_wise_tictoc:
            fprint(f"special tracking update: Finished processing sample number {nidx}/{n}")
            toc()
            tic()
            if nidx<int(n/2):
                nidx *= 2
            else:
                nidx += int(n/2**3)
        # Add each datapoint to coreset[0][0]
        coreset = coresets[0][0]
        coreset[i] = i
        # Capture index i as 1D array to ensure X[i_array] is a 2D array
        i_array = coreset[i, np.newaxis]
        # Store kernel evaluation with all points <= i
        K[i][:(i+1)] = k(i_array, coreset[:(i+1)]) 
        # Store summed kernel inner product with all points < i
        KC[0][0,i] = np.sum(K[i][:i]) 
        # Store diagonal element, kernel(xi, xi)
        diagK[i] = K[i][i] 

        # If 2^(j+1) divides (i+1), add a point from coreset[j][j2] to each of
        # coreset[j+1][2*j2] and coreset[j+1][2*j2+1]
        for j in range(min(m, largest_power_of_two(i+1))):
            parent_coresets = coresets[j]
            child_coresets = coresets[j+1]
            parent_KC = KC[j]
            child_KC = KC[j+1]
            num_parent_coresets = parent_coresets.shape[0]
            # Consider each parent coreset in turn
            for j2 in range(num_parent_coresets):
                parent_coreset = parent_coresets[j2]
                # Number of points in parent_coreset
                parent_idx = int((i+1) / num_parent_coresets)
                # Get last two points from the parent coreset
                point1, point2 = parent_coreset[parent_idx-1], parent_coreset[parent_idx-2]
                # Compute kernel(x1, x2)
                K12 = K[point1][point2]

                # Use adaptive failure threshold
                # Compute b^2 = ||f||^2 = ||k(x1,.) - k(x2,.)||_k^2
                b_sqd = diagK[point2] + diagK[point1] - 2*K12
                # Update threshold for halving parent coreset
                # a = max(b sig sqrt(log_multiplier), b^2)
                thresh = max(np.sqrt(sig_sqd[j][j2]*b_sqd*log_multiplier), b_sqd)
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
                # Note that KC[j, point2] = <k(coreset[j][l],.), k(x2, .)> and
                # KC[j, point1] = <k(coreset[j][l],.), k(x1, .)> + k(x1,x2)         
                if parent_idx > 2:
                    #parent_points = coresets[j][:(parent_idx-2)]
                    alpha = parent_KC[j2, parent_idx-1] - parent_KC[j2, parent_idx-2] - K12 
                else:
                    alpha = 0.
                # Identify the two new child coresets
                left_child_coreset = child_coresets[2*j2]
                right_child_coreset = child_coresets[2*j2+1]
                # Number of points in each new coreset
                child_idx = int(parent_idx/2-1)
                if child_idx > 0:
                    # Subtract 2 * inner product with all points in left child coreset:
                    # - 2 * sum_{l < child_idx} <k(coreset[j+1][l],.), k(x1, .) - k(x2, .)> 
                    child_points = left_child_coreset[:child_idx]
                    point1_kernel_sum = np.sum(K[point1][child_points])
                    point2_kernel_sum = np.sum(K[point2][child_points]) 
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
    """
    Select the candidate coreset with smallest MMD to all points in X (after comparing with
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
    """
    # Compute meanK if appropriate
    if meanK is None and not store_K:
        meanK = kernel_matrix_row_mean(X, kernel)
        
    # Return refined version of best coreset
    return(refine(X, best(X, coresets, kernel, store_K=store_K, meanK=meanK), kernel, meanK=meanK, unique=unique))

def best(X, coresets, kernel, store_K=False, meanK=None):
    """
    Select the candidate coreset with smallest MMD to all points in X (after comparing with
    a baseline standard thinning coreset).
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      coresets: 2D array with each row specifying the row indices of X belonging to a coreset
      kernel: kernel(x, X) (typically the target kernel, k);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      store_K: If False, runs O(nd) space version which does not store kernel
        matrix; if True, stores n x n kernel matrix
      meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
        used to speed up computation when not None
    """
    n = X.shape[0]
    if store_K:
        # Store matrix of kernel evaluations between each pair of points
        K = np.empty((n,n))
        for ii in range(n):
            K[ii] = kernel(X[ii, np.newaxis], X)
            
    # Compute the relative MMD^2 of a standard thinning coreset
    # RelMMD^2 = MMD^2 - np.mean(K) = np.mean(K[coreset][:,coreset]) - 2*np.mean(K[coreset, :]))
    coreset_size = coresets.shape[1]
    # initialize as a standard thinned coreset from the end
    best_coreset = np.array(range(n-1,-1,-int(n/coreset_size)))[::-1] 
    best_rel_mmd2 = (squared_emp_rel_mmd_K(K, best_coreset) if store_K
                     else squared_emp_rel_mmd_X(X, best_coreset, kernel, meanK))
    
    # Select the better of standard thinning coreset and the best input coreset
    for coreset in coresets:
        rel_mmd2 = (squared_emp_rel_mmd_K(K, coreset) if store_K
                    else squared_emp_rel_mmd_X(X, coreset, kernel, meanK))
        if rel_mmd2 < best_rel_mmd2:
            best_rel_mmd2 = rel_mmd2
            best_coreset = coreset
    
    return(best_coreset)

def refine(X, coreset, kernel, meanK=None, unique=False):
    """
    Replace each element of a coreset in turn by the point in X that yields the minimum 
    MMD between all points in X and the resulting coreset.
    
    Args:
      X: Input sequence of sample points with shape (n, d)
      coreset: Row indices of X representing coreset
      kernel: Kernel function (typically the target kernel, k);
        kernel(y,X) returns array of kernel evaluations between y and each row of X
      meanK: None or array of length n with meanK[ii] = mean of kernel(X[ii], X);
        used to speed up computation when not None
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
    if meanK is None:
        sufficient_stat = np.empty(n)
        for ii in range(n):
            # Kernel evaluation between xi and every row in X
            kii =  kernel(X[ii, np.newaxis], X)
            sufficient_stat[ii] = 2*(np.mean(kii[coreset])-np.mean(kii)) + kii[ii]/coreset_size    
    else:
        # Initialize to kernel diagonal normalized by coreset_size - 2 * meanK
        sufficient_stat = kernel(X, X)/coreset_size - 2 * meanK
        # Add in contribution of coreset
        for ii in range(n):
            # Kernel evaluation between xi and every coreset row in X
            kiicore =  kernel(X[ii, np.newaxis], X[coreset])
            sufficient_stat[ii] += 2*np.mean(kiicore)
    
    # Consider each coreset point in turn 
    for coreset_idx in range(coreset_size):
        # Remove the contribution of this point from the normalized coreset sum in sufficient stat
        sufficient_stat -= kernel(X[coreset[coreset_idx], np.newaxis], X)*two_over_coreset_size
        # Find the input point that would reduce MMD the most
        best_point = np.argmin(sufficient_stat)
        # Add best point to coreset and its contribution to sufficient stat
        coreset[coreset_idx] = best_point
        sufficient_stat += kernel(X[best_point, np.newaxis], X)*two_over_coreset_size
        if unique:
            sufficient_stat[best_point] = np.inf
            
    return(coreset)

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
        meanK[ii] =  np.mean(kernel(X[ii, np.newaxis], X))
    return(meanK)

def squared_emp_rel_mmd_K(K, coreset): 
    """Computes squared empirical relative MMD between a distribution weighted equally 
    on all points and one weighted equally on the indices in coreset.
    RelMMD^2 = MMD^2 - np.mean(K) = np.mean(K[coreset][:,coreset]) - 2*np.mean(K[coreset, :]))
    
    Args:
      K: Matrix of pairwise kernel evaluations
      coreset: Row indices of K representing coreset
    """
    # Note: K[coreset][:,coreset] returns the appropriate coreset x coreset submatrix 
    # while K[coreset, coreset] does not (it returns [K[i,i] for i in coreset])
    return(np.mean(K[coreset][:,coreset]) - 2*np.mean(K[coreset, :]))

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
            kii = kernel(X[ii, np.newaxis], X)
            k_core_core += np.mean(kii[coreset])
            k_core_all += np.mean(kii)
    else:
        k_core_all = np.sum(meanK[coreset])
        for ii in coreset:
            # Add mean kernel evaluation between x_{ii} and all coreset points in X
            k_core_core += np.mean(kernel(X[ii, np.newaxis], X[coreset]))
        
    # Return mean(K[coreset][:,coreset]) - 2 * mean(K[coreset,:])
    coreset_size = len(coreset)
    return((k_core_core - 2*k_core_all)/coreset_size)
