"""sobolev kernel smoothness functionality.

Cython implementation of functions involving sobolev kernel smoothness evaluation.
"""
import numpy as np
cimport numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport cython
from libc.math cimport sqrt, log, exp, cos
from libc.stdlib cimport rand, RAND_MAX, srand
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.stdio cimport printf
# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. 
np.import_array()

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Sobolev Kernel Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double single_sobolev_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const long s) nogil:
    """
    Computes a single Sobolev kernel k(X1-X2, s) 
    between two points X1 and X2
    
    Args:
      X1: array of size d
      X2: array of size d
      s: sobolev kernel smoothness
    """
    cdef long d = X1.shape[0]
    cdef double x, ans
    cdef long j
    
    cdef double pi_factor

    # Compute the squared Euclidean distance between X1 and X2
    if s == 1:
        pi_factor = pi ** 2
    if s == 2:
        pi_factor = pi ** 4
    if s == 3:
        pi_factor = pi ** 6

    ans = 1.
    for j in range(d):
        x = X1[j]-X2[j]
        if x <0:
            x += 1
        if s == 1:
            ans *= (1. + 2. * pi_factor * ((x ** 2) - x + 1. / 6.))
        if s == 2:
            ans = ans * ( 1. - pi_factor * 2. / 3. * \
                ((x ** 4) - 2. * (x ** 3) + (x ** 2) - 1. / 30.))
        if s == 3:
            ans = ans * ( 1 + pi_factor * 4. / 45. * ((x**6) - 3 * (x**5) + \
                    5. / 2. * (x**4) - (x ** 2) / 2. + 1. / 42.))
    return(ans)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double sobolev_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const long[:] s) nogil:
    """
    Computes a sum of Sobolev kernels sum_j k(X1-X2, s_j) 
    between two points X1 and X2
    
    Args:
      X1: array of size d
      X2: array of size d
      s: array of sobolev kernel smoothness
    """
    
    cdef long d = X1.shape[0]
    cdef long num_kernels = s.shape[0]
    
    cdef long j
    cdef double kernel_sum
    
    # Compute the kernel sum
    kernel_sum = single_sobolev_kernel_two_points(X1, X2, s[0])
    for j in range(1, num_kernels):
        kernel_sum += single_sobolev_kernel_two_points(X1, X2, s[j])
    return(kernel_sum)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sobolev_kernel(const double[:,:] X1,
                           const double[:,:] X2,
                           const long s,
                           double[:,:] K) nogil:
    """
    Computes the Sobolev kernel matrix between each rows of X1 and each rows of X2
    and stores in K
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
      K: Matrix of size (n1,n2) to store kernel matrix
    """
    
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    
    
    for i in range(n1):
        for j in range(n2):
            K[i,j] = single_sobolev_kernel_two_points(X1[i], X2[j], s)
                
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sobolev_kernel_same(const double[:,:] X1,
                                const long s,
                                double[:,:] K) nogil:
    """
    Computes the sobolev kernel smoothness matrix between each pair of rows in X1
    and stores in K
    
    Args:
      X1: Matrix of size (n1,d)
      s: sobolev kernel smoothness
      K: Empty matrix of size (n1,n1) to store kernel matrix
    """
    
    cdef long n1 = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    
    
    for i in range(n1):
        for j in range(i+1):
            K[i,j] = single_sobolev_kernel_two_points(X1[i], X1[j], s)
            if j<i:
                K[j,i] = K[i,j]
            
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sobolev_kernel_by_row(const double[:] X1,
                                  const double[:,:] X2,
                                  const long s,
                                  double[:] K) nogil:
    """
    Computes the sobolev kernel matrix between X1 and each row of X2
    and stores in K
    
    Args:
      X1: Vector of size d
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
      K: Vector of size n2 to store kernel values
    """
    
    cdef long n2 = X2.shape[0]
    cdef long d = X2.shape[1]
    
    cdef long i
    
    for i in range(n2):
        K[i] = single_sobolev_kernel_two_points(X1, X2[i], s)
   
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_sobolev_kernel(const double[:,:] X1,
                                 const double[:,:] X2,
                                 const long s) nogil:
    """
    Returns the sum of sobolev kernel evaluations between each row of X1 
    and each row of X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    
    for i in range(n1):
        for j in range(n2):
            total_sum += single_sobolev_kernel_two_points(X1[i], X2[j], s)
            
    return(total_sum)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_sobolev_kernel_same(const double[:,:] X1,
                                      const long s) nogil:
    """
    Returns the sum of sobolev kernel evaluations between each pair of 
    rows of X1
    
    Args:
      X1: Matrix of size (n1,d)
      s: sobolev kernel smoothness
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    
    for i in range(n1):
        for j in range(i+1):
            if j < i:    
                total_sum += 2*single_sobolev_kernel_two_points(X1[i], X1[j], s)
            else:
                total_sum += single_sobolev_kernel_two_points(X1[i], X1[j], s)
            
    return(total_sum)        
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_sobolev_kernel_linear_eval(const double[:,:] X1,
                                             const double[:,:] X2,
                                             const long s) nogil:
    """
    Computes the sum of sobolev kernel smoothness of order s evaluations between the 
    i-th row of X1 and the i-th row of X2; they need to have same number of rows
    
    Args:
      X1: Matrix of size (n,d)
      X2: Matrix of size (n,d)
      s: sobolev kernel smoothness
    """
    
    cdef double total_sum = 0
    cdef long n = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i
    
    for i in range(n):
        total_sum += single_sobolev_kernel_two_points(X1[i], X2[i], s)
            
    return(total_sum)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_sobolev_kernel_by_bin(const double[:,:] X1,
                                      const double[:,:] X2,
                                      const long s,
                                      double[:,:] K_sum) nogil:
    """
    Partitions the rows of X1 and the rows of X2 into bins of size 
    bin_size = (n1+n2) // num_bins, computes sum_sobolev_kernel for 
    each pair of bins, and stores the result in K_sum
    
    Note: Assumes that bin_size evenly divides n1 and n2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
      K_sum: Matrix of size (num_bins,num_bins) to store kernel sums
    """
    
    cdef long num_bins = K_sum.shape[0]
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long bin_size = (n1+n2) // num_bins
    cdef long num_bins1 = n1 // bin_size
    cdef long num_bins2 = n2 // bin_size

    # Compute kernel sum for each pair of X1 bins
    cdef const double[:,:] bin1, bin2
    cdef long k1, k2
    cdef long bin1_start = 0
    cdef long bin1_end = bin_size
    cdef long bin2_start
    cdef long bin2_end
    for k1 in range(num_bins1):
        bin1 = X1[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(k1):
            bin2 = X1[bin2_start:bin2_end,:]
            K_sum[k1,k2] = sum_sobolev_kernel(bin1,bin2,s)
            K_sum[k2,k1] = K_sum[k1,k2]
            bin2_start = bin2_end
            bin2_end += bin_size
        K_sum[k1,k1] = sum_sobolev_kernel_same(bin1,s)
        bin1_start = bin1_end
        bin1_end += bin_size
                    
    # Compute kernel sum for each pair of X2 bins
    bin1_start = 0
    bin1_end = bin_size
    for k1 in range(num_bins2):
        bin1 = X2[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(k1):
            bin2 = X2[bin2_start:bin2_end,:]
            K_sum[num_bins1+k1,num_bins1+k2] = sum_sobolev_kernel(bin1,bin2,s)
            K_sum[num_bins1+k2,num_bins1+k1] = K_sum[num_bins1+k1,num_bins1+k2]
            bin2_start = bin2_end
            bin2_end += bin_size
        K_sum[num_bins1+k1,num_bins1+k1] = sum_sobolev_kernel_same(bin1,s)
        bin1_start = bin1_end
        bin1_end += bin_size
                    
    # Compute kernel sum between each pairing of one X1 and one X2 coreset
    bin1_start = 0
    bin1_end = bin_size
    for k1 in range(num_bins1):
        bin1 = X1[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(num_bins2):
            bin2 = X2[bin2_start:bin2_end,:]
            K_sum[k1,num_bins1+k2] = sum_sobolev_kernel(bin1,bin2,s)
            K_sum[num_bins1+k2,k1] = K_sum[k1,num_bins1+k2]
            bin2_start = bin2_end
            bin2_end += bin_size
        bin1_start = bin1_end
        bin1_end += bin_size
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_sobolev_kernel_by_bin_aggregated(const double[:,:] X1,
                                                 const double[:,:] X2,
                                                 const long[:] s,
                                                 double[:,:,:] K_sum) nogil:
    """
    Partitions the rows of X1 and the rows of X2 into bins of size 
    bin_size = (n1+n2) // num_bins, computes sum_sobolev_kernel for 
    each pair of bins and each smoothness in s, and stores the result 
    in K_sum. 
    
    Note: Only populates the lower-triangular entries K_sum[k1,k2,:] 
          for k1 >= k2.
    
    Note: Assumes that bin_size evenly divides n1 and n2.
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: Length l vector of kernel smoothness  for the sobolev kernels 
      K_sum: Zeros matrix of size (num_bins,num_bins,l) to store kernel sums
    """
    
    cdef long num_bins = K_sum.shape[0]
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long bin_size = (n1+n2) // num_bins
    cdef long num_bins1 = n1 // bin_size
    cdef long num_bins2 = n2 // bin_size
    
    # Compute kernel sums for each pair of X1 bins
    cdef const double[:,:] bin1, bin2
    cdef long k1, k2
    cdef long bin1_start = 0
    cdef long bin1_end = bin_size
    cdef long bin2_start
    cdef long bin2_end
    for k1 in range(num_bins1):
        bin1 = X1[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(k1):
            bin2 = X1[bin2_start:bin2_end,:]
            sum_sobolev_kernel_aggregated(bin1,bin2,s,K_sum[k1,k2,:])
            bin2_start = bin2_end
            bin2_end += bin_size
        sum_sobolev_kernel_same_aggregated(bin1,s,K_sum[k1,k1,:])
        bin1_start = bin1_end
        bin1_end += bin_size
                    
    # Compute kernel sums for each pair of X2 bins
    bin1_start = 0
    bin1_end = bin_size
    for k1 in range(num_bins2):
        bin1 = X2[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(k1):
            bin2 = X2[bin2_start:bin2_end,:]
            sum_sobolev_kernel_aggregated(bin1,bin2,s,K_sum[num_bins1+k1,num_bins1+k2,:])
            bin2_start = bin2_end
            bin2_end += bin_size
        sum_sobolev_kernel_same_aggregated(bin1,s,K_sum[num_bins1+k1,num_bins1+k1,:])
        bin1_start = bin1_end
        bin1_end += bin_size
                    
    # Compute kernel sums between each pairing of one X1 and one X2 coreset
    bin1_start = 0
    bin1_end = bin_size
    for k1 in range(num_bins1):
        bin1 = X1[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(num_bins2):
            bin2 = X2[bin2_start:bin2_end,:]
            sum_sobolev_kernel_aggregated(bin1,bin2,s,K_sum[num_bins1+k2,k1,:])
            bin2_start = bin2_end
            bin2_end += bin_size
        bin1_start = bin1_end
        bin1_end += bin_size

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_sobolev_kernel_aggregated(const double[:,:] X1,
                                          const double[:,:] X2,
                                          const long[:] s,
                                          double[:] results) nogil:
    """
    Computes the sum of sobolev kernel smoothness of order s evaluations between all rows of X1 and all rows of X2, 
    for all the kernel smoothness in s and stores in results.
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: Length l vector of kernel smoothness  for the sobolev kernels 
      results: Zeros vector of length L in which results will be stored
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j, l
    
    for i in range(n1):
        for j in range(n2):
            for l in range(len(s)):
                results[l] += single_sobolev_kernel_two_points(X1[i], X2[j], s[l]) 
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_sobolev_kernel_same_aggregated(const double[:,:] X1,
                                               const long[:] s,
                                               double[:] results) nogil:
    """
    Computes the sum of sobolev kernel smoothness of order s evaluations between all rows of X1 and all rows of X1,
    for all the kernel smoothness in s and stores in results.
    
    Args:
      X1: Matrix of size (n1,d)
      s: Length l vector of kernel smoothness  for the sobolev kernels 
      results: Zeros vector of length L in which results will be stored
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j, k, l
    
    for i in range(n1):
        for j in range(i+1):
            if j < i:
                for l in range(len(s)):
                    results[l] += 2* single_sobolev_kernel_two_points(X1[i], X2[j], s[l]) 
            else:
                for l in range(len(s)):
                    results[l] += single_sobolev_kernel_two_points(X1[i], X2[j], s[l]) 

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double biased_sqMMD_sobolev(const double[:,:] X1,
                                   const double[:,:] X2,
                                   const long s) nogil:
    """
    Computes the biased quadratic squared MMD estimator for the sobolev kernel smoothness of order s from samples X1 and X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n12,d)
      s: sobolev kernel smoothness
    """
    
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef double first_term = sum_sobolev_kernel(X1,X1,s)
    cdef double second_term = sum_sobolev_kernel(X1,X2,s)
    cdef double third_term = sum_sobolev_kernel(X2,X2,s)
    cdef double bsqMMD = first_term/(n1*n1) - 2*second_term/(n1*n2) + third_term/(n2*n2)
            
    return(bsqMMD)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double unbiased_sqMMD_sobolev(const double[:,:] X1,
                                     const double[:,:] X2,
                                     const long s) nogil:
    """
    Computes the unbiased quadratic squared MMD estimator for the sobolev kernel smoothness of order s from samples X1 and X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
    """
    
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef double first_term = sum_sobolev_kernel_same(X1,s)
    cdef double second_term = sum_sobolev_kernel(X1,X2,s)
    cdef double third_term = sum_sobolev_kernel_same(X2,s)
    cdef double extra_terms_first = sum_sobolev_kernel_linear_eval(X1,X1,s)
    cdef double extra_terms_third = sum_sobolev_kernel_linear_eval(X2,X2,s)
    cdef double usqMMD = (first_term-extra_terms_first)/(n1*(n1-1)) - 2*second_term/(n1*n2) + (third_term-extra_terms_third)/(n2*(n2-1))
            
    return(usqMMD)
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void block_sqMMD_sobolev(const double[:,:] X1,
                                const double[:,:] X2,
                                const long s,
                                const long block_size,
                                double[:] results) nogil:
    """Computes the block quadratic squared MMD estimator for the sobolev kernel smoothness of order s from samples X1 and X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n12,d)
      s: sobolev kernel smoothness
      block_size: long, size of blocks
    """

    cdef long n = X1.shape[0]
    # Compute ceil(n/block_size)
    cdef long n_splits = n//block_size + (n % block_size != 0)
    cdef long first_index
    cdef long last_index
    cdef double results_1

    cdef double total_sum = 0
    cdef double total_sum_sqd = 0
    cdef double split_sum
    cdef long i
    for i in range(n_splits):
        split_sum = 0
        first_index = i*block_size
        if i != n_splits - 1:
            last_index = (i+1)*block_size
        else:
            last_index = n
        for j in range(first_index,last_index):
            for k in range(first_index,j): #we only need to look at the lower triangular matrix
                h_value = h_sobolev(X1[j],X1[k],X2[j],X2[k],s)
                split_sum += h_value
        split_sum = 2*split_sum/(1.0*(last_index-first_index))/(1.0*(last_index-first_index-1))
        total_sum += split_sum
        total_sum_sqd += split_sum**2
    results[0] = total_sum/n_splits
    if n_splits != 1:
        #results_1 = (total_sum_sqd/n_splits - (total_sum/n_splits)**2*((n_splits+1)/(n_splits-1)))/n_splits ##we use an unbiased estimate of the variance of the block values
        results_1 = (total_sum_sqd/n_splits - (total_sum/n_splits)**2)/(n_splits-1)
        if results_1 <= 0:
            results[1] = 0
        else:
            results[1] = results_1
    else:
        results[1] = 0
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void block_sqMMD_sobolev_reordered(const double[:,:] X1,
                                          const double[:,:] X2,
                                          const long s,
                                          const long block_size,
                                          const long seed,
                                          long[:] epsilon,
                                          double[:] results) nogil:
    """Computes the block squared MMD estimator for the sobolev kernel smoothness of order s from samples X1 and X2, reordering before constructing blocks
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n12,d)
      s: sobolev kernel smoothness
      block_size: long, size of blocks
      seed: random seed
      epsilon: used to store random vector
    """
    #printf('epsilon cython: %u, %u\n', epsilon[0], epsilon[1])
    
    cdef long n = X1.shape[0]
    # Compute ceil(n/block_size)
    cdef long n_splits = n//block_size + (n % block_size != 0)
    cdef long first_index
    cdef long last_index
    cdef double results_2, results_3

    cdef double total_sum = 0
    cdef double total_sum_sqd = 0
    cdef double total_sum_epsilon = 0
    cdef double total_sum_sqd_epsilon = 0
    cdef double split_sum
    cdef double split_sum_epsilon
    cdef double h_value
    cdef long i, j, k
    
    srand(seed)
    
    for i in range(n_splits):
        srand(i)
        for j in range(block_size):
            epsilon[j] = rand() % 2
        split_sum = 0
        split_sum_epsilon = 0 
        first_index = i*block_size
        if i != n_splits - 1:
            last_index = (i+1)*block_size
        else:
            last_index = n
        for j in range(first_index,last_index):
            for k in range(first_index,j): #we only need to look at the lower triangular matrix
                h_value = h_sobolev(X1[j],X1[k],X2[j],X2[k],s)
                split_sum += h_value
                split_sum_epsilon += (2.0*epsilon[j-first_index]-1.0)*(2.0*epsilon[k-first_index]-1.0)*h_value
        split_sum = 2*split_sum/(1.0*(last_index-first_index)*(last_index-first_index-1))
        split_sum_epsilon = 2*split_sum_epsilon/(1.0*(last_index-first_index)*(last_index-first_index-1))
        total_sum += split_sum
        total_sum_epsilon += split_sum_epsilon
        total_sum_sqd += split_sum**2
        total_sum_sqd_epsilon += split_sum_epsilon**2
    results[0] = total_sum/(n_splits*1.0)
    results[1] = total_sum_epsilon/(n_splits*1.0)
    if n_splits != 1:
        results_2 = (total_sum_sqd/(n_splits*1.0) - (total_sum/(1.0*n_splits))**2)/(1.0*n_splits-1) ##we use an unbiased estimate of the variance of the block values 
        if results_2 <= 0:
            results[2] = 0
        else:
            results[2] = results_2
        results_3 = (total_sum_sqd_epsilon/(n_splits*1.0) - (total_sum_epsilon/(1.0*n_splits))**2)/(1.0*n_splits-1)
        if results_3 <= 0:
            results[3] = 0
        else:
            results[3] = results_3
    else:
        results[2] = 0
        results[3] = 0
        
    printf('n_splits: %u, results[0]: %.15f, results[1]: %.15f, results[2]: %.15f, results[3]: %.15f\n', n_splits, results[0], results[1], results[2], results[3])

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double block_sqMMD_sobolev_Rademacher(const double[:,:] X1,
                                             const double[:,:] X2,
                                             const long s,
                                             const long block_size,
                                             const long long[:,:] epsilon,
                                             double[:] split_values,
                                             double[:] sqMMD_values) nogil:
    """Computes the block squared MMD estimator for the sobolev kernel smoothness of order s from samples X1 and X2, 
    for different Rademacher vectors
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
      block_size: long, size of blocks
      epsilon: used to store random vector
      split_values: used in the procedure
      sqMMD_values: used to store results
    """
    cdef long n = X1.shape[0]
    cdef long n_splits = n//block_size + (n % block_size != 0)
    cdef long B = epsilon.shape[1]
    
    cdef long first_index, last_index
    cdef long i, j, k, l, m
    cdef long long eps_j, eps_k
    cdef timespec ts
    cdef double start, end
    
    cdef double h_value, time
    
    #Record start time
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    
    for i in range(n_splits):
        
        #printf('Split %u of %u\n', i, n_splits)
        for j in range(len(split_values)):
            split_values[j] = 0
        first_index = i*block_size
        if i != n_splits - 1:
            last_index = (i+1)*block_size
        else:
            last_index = n
        
        for j in range(first_index,last_index):
            for k in range(first_index,j): #we only need to look at the lower triangular matrix
                h_value = h_sobolev(X1[j],X1[k],X2[j],X2[k],s)
                split_values[B] += h_value
                for l in range(B):
                    split_values[l] += epsilon[j,l]*epsilon[k,l]*h_value
        
        for j in range(len(split_values)):
            split_values[j] = 2*split_values[j]/(1.0*(last_index-first_index)*(last_index-first_index-1)*n_splits)
            sqMMD_values[j] += split_values[j]
     
    clock_gettime(CLOCK_REALTIME, &ts)
    end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    time = end-start
    
    printf('Test statistic value %f in time %f. n_splits: %u, block_size: %u\n', sqMMD_values[B], time, n_splits, block_size)
    
    return time

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double h_sobolev(const double[:] x1,
                        const double[:] x2,
                        const double[:] y1,
                        const double[:] y2,
                        const long s) nogil:
    """
    Computes the kernel function h((x1,y1),(x2,y2)) = k(x1,x1) + k(y2,y2) - k(x1,y2) - k(x2,y1), where 
    k is the sobolev kernel smoothness of order s
    
    Args:
      x1: vector of size d
      x2: vector of size d
      y1: vector of size d
      y2: vector of size d
      s: sobolev kernel smoothness
    """
    
    cdef double ans = single_sobolev_kernel_two_points(x1, x2, s)
    ans += single_sobolev_kernel_two_points(y1, y2, s)
    ans -= single_sobolev_kernel_two_points(x1, y2, s)
    ans -= single_sobolev_kernel_two_points(x2, y1, s)
    return(ans)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void h_sobolev_aggregated(const double[:] x1,
                                 const double[:] x2,
                                 const double[:] y1,
                                 const double[:] y2,
                                 const long[:] s,
                                 double[:] h_values) nogil:
    """
    Computes the kernel function h((x1,y1),(x2,y2)) = k(x1,x1) + k(y2,y2) - k(x1,y2) - k(x2,y1), where k is the sobolev kernel smoothness of order s,
    for all the kernel smoothness in s
    
    Args:
      x1: vector of size d
      x2: vector of size d
      y1: vector of size d
      y2: vector of size d
      s: sobolev kernel smoothness
      h_values: used to store results
    """

    cdef long k 
    for k in range(len(s)):
        h_values[k] = single_sobolev_kernel_two_points(x1, x2, s[k])
        h_values[k] += single_sobolev_kernel_two_points(y1, y2, s[k])
        h_values[k] -= single_sobolev_kernel_two_points(x1, y2, s[k])
        h_values[k] -= single_sobolev_kernel_two_points(x2, y1, s[k])


@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void incomplete_sqMMD_sobolev(const double[:,:] X1,
                                     const double[:,:] X2,
                                     const long long[:] incomplete_list,
                                     const long long max_l,
                                     const long s,
                                     const long seed,
                                     double[:] sqMMD_list,
                                     double[:] sqMMD_variances,
                                     double[:] sqMMD_times) nogil:
    """
    Computes the incomplete squared MMD estimator
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      incomplete_list: vector containing numbers of pairs, ordered
      max_l: largest number of pairs (largest/last element of incomplete_list)
      s: sobolev kernel smoothness
      seed: random seed
      sqMMD_list: used to store results
      sqMMD_variances: used to store results
      sqMMD_times: used to store results
    """
    
    cdef long n_samples = X1.shape[0]
    cdef double total_value = 0
    cdef double total_value_sqd = 0
    cdef double h_value
    cdef long incomplete_index = 0
    cdef long D0, D1
    cdef long long j
    cdef timespec ts
    cdef double start, end
    
    #Record start time
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    
    #Set seed for rand()
    srand(seed)
    
    for j in range(max_l):
  
        #Sample pair
        D0 = rand() % n_samples 
        D1 = (D0 + 1 + rand() % (n_samples-1))%n_samples 
        
        #Compute contribution of pair
        h_value = h_sobolev(X1[D0],X1[D1],X2[D0],X2[D1],s)
        total_value += h_value
        total_value_sqd += h_value**2
        
        #If needed, record sqMMD value and time for corresponding incomplete size
        if j == incomplete_list[incomplete_index] - 1:
            sqMMD_list[incomplete_index] = total_value/((j+1)*1.0)
            sqMMD_variances[incomplete_index] = (total_value_sqd/((j+1)*1.0) - (total_value**2/((j+1)*1.0))/((j+1)*1.0))/((j+1)*1.0) 
            clock_gettime(CLOCK_REALTIME, &ts)
            end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
            sqMMD_times[incomplete_index] = end-start
            incomplete_index += 1
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void incomplete_sqMMD_sobolev_Rademacher_subdiagonals(const double[:,:] X1,
                                                             const double[:,:] X2,
                                                             const long long[:] incomplete_list,
                                                             const long B,
                                                             const long long max_l,
                                                             const long s,
                                                             const long[:,:] epsilon,
                                                             double[:,:] sqMMD_matrix,
                                                             double[:] sqMMD_vector,
                                                             double[:] sqMMD_times) nogil:
    """
    Computes the incomplete squared MMD estimator for the sobolev kernel smoothness of order s from samples X1 and X2, 
    for different Rademacher vectors
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      incomplete_list: vector containing numbers of pairs, ordered
      B: number of Rademacher vectors
      max_l: largest number of pairs (largest/last element of incomplete_list)
      s: sobolev kernel smoothness
      epsilon: used to store Rademacher vector
      sqMMD_list: used to store results
      sqMMD_variances: used to store results
      sqMMD_times: used to store results
    """
    
    cdef long n_samples = X1.shape[0]
    cdef double h_value
    cdef long incomplete_index = 0
    cdef long long j

    cdef long k, l, r
    cdef long D0, D1

    cdef long long eps_D0, eps_D1
    cdef timespec ts
    cdef double start, end
    
    #Record start time
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    
    printf('Run ktc.incomplete_sqMMD_sobolev_Rademacher inside\n')
    
    D0 = 0
    D1 = 1
    r = 1
    for j in range(max_l):
        if r > n_samples - 1:
            printf('Error: r > n_samples - 1: r = %u, n_samples = %u, D0 = %u, D1 = %u \n', r, n_samples, D0, D1)
        
        #Compute contribution of pair
        h_value = h_sobolev(X1[D0],X1[D1],X2[D0],X2[D1],s)
        
        #Update sqMMD vector
        for l in range(B):
            sqMMD_vector[l] += epsilon[D0,l]*epsilon[D1,l]*h_value/n_samples
        sqMMD_vector[B] += h_value/n_samples
        
        #If j is equal to some element of incomplete_list (minus 1), store sqMMD values and times    
        if j == incomplete_list[incomplete_index] - 1:
            printf('incomplete_index %u of %u: %u\n', incomplete_index+1, len(incomplete_list), incomplete_list[incomplete_index])
            for l in range(B+1):
                sqMMD_matrix[incomplete_index,l] = sqMMD_vector[l]
            clock_gettime(CLOCK_REALTIME, &ts)
            end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
            sqMMD_times[incomplete_index] = end-start
            incomplete_index += 1
            
        if D0 == n_samples - r - 1:
            r += 1
            D0 = 0
            D1 = r
        else:
            D0 += 1
            D1 += 1
    
    for j in range(incomplete_index,len(incomplete_list)):
        for l in range(B+1):
            sqMMD_matrix[j,l] = sqMMD_vector[l]
        sqMMD_times[j] = end-start
            
    printf('final incomplete_index: %u. len(incomplete_list): %u.\n', incomplete_index, len(incomplete_list))
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void incomplete_sqMMD_sobolev_Rademacher_aggregated(const double[:,:] X1,
                                                           const double[:,:] X2,
                                                           const long long[:] incomplete_list,
                                                           const long B,
                                                           const long long max_l,
                                                           const long[:] s,
                                                           const long[:,:] epsilon,
                                                           double[:,:,:] sqMMD_matrix,
                                                           double[:,:] sqMMD_vector,
                                                           double[:] sqMMD_times,
                                                           double[:] h_values) nogil:
    """
    Computes the incomplete squared MMD estimator for the sobolev kernel smoothness of order s from samples X1 and X2, 
    for different Rademacher vectors and different smoothness
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      incomplete_list: vector containing numbers of pairs, ordered
      max_l: largest number of pairs (largest/last element of incomplete_list)
      epsilon: used to store Rademacher vector
      s: sobolev kernel smoothness
      epsilon: random seed
      sqMMD_list: used to store results
      sqMMD_variances: used to store results
      sqMMD_times: used to store results
    """
    
    cdef long n_samples = X1.shape[0]
    cdef double h_value
    cdef long incomplete_index = 0
    cdef long long j

    cdef long k, l, r, i
    cdef long D0, D1

    cdef long long eps_D0, eps_D1
    cdef timespec ts
    cdef double start, end
    
    #Record start time
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    
    printf('Run ktc.incomplete_sqMMD_sobolev_Rademacher inside\n')
    
    D0 = 0
    D1 = 1
    r = 1
    for j in range(max_l):
        if r > n_samples - 1:
            printf('Error: r > n_samples - 1: r = %u, n_samples = %u, D0 = %u, D1 = %u \n', r, n_samples, D0, D1)
        
        #Compute contribution of pair
        h_sobolev_aggregated(X1[D0],X1[D1],X2[D0],X2[D1],s,h_values)
        
        #Update sqMMD vector
        for i in range(len(s)):
            for l in range(B):
                sqMMD_vector[i,l] += epsilon[D0,l]*epsilon[D1,l]*h_values[i]/n_samples
            sqMMD_vector[i,B] += h_values[i]/n_samples
        
        #If j is equal to some element of incomplete_list (minus 1), store sqMMD values and times    
        if j == incomplete_list[incomplete_index] - 1:
            printf('incomplete_index %u of %u: %u\n', incomplete_index+1, len(incomplete_list), incomplete_list[incomplete_index])
            for i in range(len(s)):
                for l in range(B):
                    sqMMD_matrix[incomplete_index,i,l] = sqMMD_vector[i,l]
                sqMMD_matrix[incomplete_index,i,B] = sqMMD_vector[i,B]
            clock_gettime(CLOCK_REALTIME, &ts)
            end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
            sqMMD_times[incomplete_index] = end-start
            incomplete_index += 1
            
        if D0 == n_samples - r - 1:
            r += 1
            D0 = 0
            D1 = r
        else:
            D0 += 1
            D1 += 1
    
    for j in range(incomplete_index,len(incomplete_list)):
        for i in range(len(s)):
            for l in range(B):
                sqMMD_matrix[j,i,l] = sqMMD_vector[i,l]
            sqMMD_matrix[j,i,B] = sqMMD_vector[i,B]
        sqMMD_times[j] = end-start
            
    printf('final incomplete_index: %u. len(incomplete_list): %u.\n', incomplete_index, len(incomplete_list))

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double expectation_h_1(const double[:,:] X1,
                             const double[:,:] X2,
                             const long s) nogil:
    """
    Computes E_z[(E_{z'}h(z,z'))^2]
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double total_sum = 0
    cdef double intermediate_sum
    for i in range(n_samples):
        intermediate_sum = 0
        for j in range(n_samples):
            intermediate_sum += h_sobolev(X1[i],X1[j],X2[i],X2[j],s)
        total_sum += (intermediate_sum/n_samples)**2
    return(total_sum/n_samples)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double expectation_h_2(const double[:,:] X1,
                             const double[:,:] X2,
                             const long s) nogil:
    """
    Computes E_{z,z'}[(h(z,z'))^2]
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double total_sum = 0
    for i in range(n_samples):
        for j in range(n_samples):
            total_sum += (h_sobolev(X1[i],X1[j],X2[i],X2[j],s))**2
    return(total_sum/n_samples**2)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double expectation_h_3(const double[:,:] X1,
                             const double[:,:] X2,
                             const long s) nogil:
    """
    Compute (E_{z,z'}[h(z,z')])^2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double total_sum = 0
    for i in range(n_samples):
        for j in range(n_samples):
            total_sum += h_sobolev(X1[i],X1[j],X2[i],X2[j],s)
    return (total_sum/n_samples**2)**2

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sigma_1_sqd(const double[:,:] X1,
                         const double[:,:] X2,
                         const long s) nogil:
    """
    Compute sigma_1^2 = E_z[(E_{z'}h(z,z'))^2] - (E_{z,z'}[h(z,z')])^2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double first_term = 0
    cdef double second_term = 0
    cdef double intermediate_sum
    cdef double h_value
    for i in range(n_samples):
        intermediate_sum = 0
        for j in range(n_samples):
            h_value = h_sobolev(X1[i],X1[j],X2[i],X2[j],s)
            intermediate_sum += h_value
            second_term += h_value
        first_term += (intermediate_sum/n_samples)**2
    return (first_term/n_samples) - (second_term/n_samples**2)**2

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sigma_2_sqd(const double[:,:] X1,
                         const double[:,:] X2,
                         const long s) nogil:
    """
    Compute sigma_2^2 = E_{z,z'}[h(z,z')^2] - E_z[(E_{z'}h(z,z'))^2]
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      s: sobolev kernel smoothness
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double first_term = 0
    cdef double second_term = 0
    cdef double intermediate_sum
    cdef double h_value
    for i in range(n_samples):
        intermediate_sum = 0
        for j in range(n_samples):
            h_value = h_sobolev(X1[i],X1[j],X2[i],X2[j],s)
            intermediate_sum += h_value
            first_term += h_value**2
        second_term += (intermediate_sum/n_samples)**2
    return first_term/n_samples**2 - (second_term/n_samples)
