"""Gaussian kernel functionality.

Cython implementation of functions involving Gaussian kernel evaluation.
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian Kernel Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double gaussian_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const double[:] lam_sqd) noexcept nogil:
    """
    Computes a sum of Gaussian kernels sum_j exp(-||X1-X2||_2^2/lam_sqd[j]) 
    between two points X1 and X2
    
    Args:
      X1: array of size d
      X2: array of size d
      lam_sqd: array of squared kernel bandwidths
    """
    
    cdef long d = X1.shape[0]
    cdef long num_kernels = lam_sqd.shape[0]
    
    cdef long j
    cdef double arg, kernel_sum
    
    # Compute the squared Euclidean distance between X1 and X2
    arg = 0
    for j in range(d):
        arg += (X1[j]-X2[j])**2
    
    # Compute the kernel sum
    kernel_sum = exp(-arg/lam_sqd[0])
    for j in range(1,num_kernels):
        kernel_sum += exp(-arg/lam_sqd[j])
    return(kernel_sum)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double gaussian_kernel_one_point(const double[:] X1,
                                      const double[:] lam_sqd) noexcept nogil:
    """
    Computes a sum of Gaussian kernels sum_j exp(-||X1-X1||_2^2/lam_sqd[j]) 
    between X1 and itself
    
    Args:
      X1: 1D array representing a data point
      lam_sqd: 1D array of squared kernel bandwidths
    """
    # sum_j exp(-||X1-X1||_2^2/lam_sqd[j]) = sum_j 1
    return(lam_sqd.shape[0])

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void gaussian_kernel(const double[:,:] X1,
                           const double[:,:] X2,
                           const double lam_sqd,
                           double[:,:] K) noexcept nogil:
    """
    Computes the Gaussian kernel matrix between each row of X1 and each row of X2
    and stores in K
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
      K: Matrix of size (n1,n2) to store kernel matrix
    """
    
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    cdef double arg
    
    for i in range(n1):
        for j in range(n2):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X2[j,k])**2
            K[i,j] = exp(-arg/lam_sqd)
                
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void gaussian_kernel_same(const double[:,:] X1,
                                const double lam_sqd,
                                double[:,:] K) noexcept nogil:
    """
    Computes the Gaussian kernel matrix between each pair of rows in X1
    and stores in K
    
    Args:
      X1: Matrix of size (n1,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
      K: Empty matrix of size (n1,n1) to store kernel matrix
    """
    
    cdef long n1 = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    cdef double arg
    
    for i in range(n1):
        K[i,i] = 1
        for j in range(i):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X1[j,k])**2
            K[i,j] = exp(-arg/lam_sqd)
            K[j,i] = K[i,j]
            
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void gaussian_kernel_by_row(const double[:] X1,
                                  const double[:,:] X2,
                                  const double lam_sqd,
                                  double[:] K) noexcept nogil:
    """
    Computes the Gaussian kernel matrix between X1 and each row of X2
    and stores in K
    
    Args:
      X1: Vector of size d
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
      K: Vector of size d to store kernel values
    """
    
    cdef long n2 = X2.shape[0]
    cdef long d = X2.shape[1]
    
    cdef long i
    cdef double arg
    for i in range(n2):
        arg = 0
        for k in range(d):
            arg += (X1[k]-X2[i,k])**2
        K[i] = exp(-arg/lam_sqd)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_gaussian_kernel(const double[:,:] X1,
                                 const double[:,:] X2,
                                 const double lam_sqd) noexcept nogil:
    """
    Returns the sum of Gaussian kernel evaluations between each row of X1 
    and each row of X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    cdef double arg
    for i in range(n1):
        for j in range(n2):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X2[j,k])**2
            total_sum += exp(-arg/lam_sqd)
            
    return(total_sum)
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_gaussian_kernel_same(const double[:,:] X1,
                                      const double lam_sqd) noexcept nogil:
    """
    Returns the sum of Gaussian kernel evaluations between each pair of 
    rows of X1
    
    Args:
      X1: Matrix of size (n1,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    cdef double arg
    for i in range(n1):
        for j in range(i+1):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X1[j,k])**2
            if j < i:    
                total_sum += 2*exp(-arg/lam_sqd)
            else:
                total_sum += exp(-arg/lam_sqd)
            
    return(total_sum)        
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_gaussian_kernel_linear_eval(const double[:,:] X1,
                                             const double[:,:] X2,
                                             const double lam_sqd) noexcept nogil:
    """
    Computes the sum of Gaussian kernel evaluations between the 
    i-th row of X1 and the i-th row of X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
    """
    
    cdef double total_sum = 0
    cdef long n = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i
    cdef double arg
    for i in range(n):
        arg = 0
        for k in range(d):
            arg += (X1[i,k]-X2[i,k])**2
        total_sum += exp(-arg/lam_sqd)
            
    return(total_sum)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_gaussian_kernel_by_bin(const double[:,:] X1,
                                      const double[:,:] X2,
                                      const double lam_sqd,
                                      double[:,:] K_sum) noexcept nogil:
    """
    Partitions the rows of X1 and the rows of X2 into bins of size 
    bin_size = (n1+n2) // num_bins, computes sum_gaussian_kernel for 
    each pair of bins, and stores the result in K_sum
    
    Note: Assumes that bin_size evenly divides n1 and n2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
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
            K_sum[k1,k2] = sum_gaussian_kernel(bin1,bin2,lam_sqd)
            K_sum[k2,k1] = K_sum[k1,k2]
            bin2_start = bin2_end
            bin2_end += bin_size
        K_sum[k1,k1] = sum_gaussian_kernel_same(bin1,lam_sqd)
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
            K_sum[num_bins1+k1,num_bins1+k2] = sum_gaussian_kernel(bin1,bin2,lam_sqd)
            K_sum[num_bins1+k2,num_bins1+k1] = K_sum[num_bins1+k1,num_bins1+k2]
            bin2_start = bin2_end
            bin2_end += bin_size
        K_sum[num_bins1+k1,num_bins1+k1] = sum_gaussian_kernel_same(bin1,lam_sqd)
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
            K_sum[k1,num_bins1+k2] = sum_gaussian_kernel(bin1,bin2,lam_sqd)
            K_sum[num_bins1+k2,k1] = K_sum[k1,num_bins1+k2]
            bin2_start = bin2_end
            bin2_end += bin_size
        bin1_start = bin1_end
        bin1_end += bin_size
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_gaussian_kernel_by_bin_aggregated(const double[:,:] X1,
                                                 const double[:,:] X2,
                                                 const double[:] lam_sqd,
                                                 double[:,:,:] K_sum) noexcept nogil:
    """
    Partitions the rows of X1 and the rows of X2 into bins of size 
    bin_size = (n1+n2) // num_bins, computes sum_gaussian_kernel for 
    each pair of bins and each squared bandwidth, and stores the result 
    in K_sum. 
    
    Note: Only populates the lower-triangular entries K_sum[k1,k2,:] 
          for k1 >= k2.
    
    Note: Assumes that bin_size evenly divides n1 and n2.
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: Length l vector of kernel squared bandwidths for the kernel 
        exp(-||x-y||_2^2/lam_sqd)
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
            sum_gaussian_kernel_aggregated(bin1,bin2,lam_sqd,K_sum[k1,k2,:])
            bin2_start = bin2_end
            bin2_end += bin_size
        sum_gaussian_kernel_same_aggregated(bin1,lam_sqd,K_sum[k1,k1,:])
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
            sum_gaussian_kernel_aggregated(bin1,bin2,lam_sqd,K_sum[num_bins1+k1,num_bins1+k2,:])
            bin2_start = bin2_end
            bin2_end += bin_size
        sum_gaussian_kernel_same_aggregated(bin1,lam_sqd,K_sum[num_bins1+k1,num_bins1+k1,:])
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
            sum_gaussian_kernel_aggregated(bin1,bin2,lam_sqd,K_sum[num_bins1+k2,k1,:])
            bin2_start = bin2_end
            bin2_end += bin_size
        bin1_start = bin1_end
        bin1_end += bin_size

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_gaussian_kernel_aggregated(const double[:,:] X1,
                                          const double[:,:] X2,
                                          const double[:] lam_sqd,
                                          double[:] results) noexcept nogil:
    """
    Computes the sum of Gaussian kernel evaluations between all rows of X1 and all rows of X2, 
    for all the squared bandwidths in lam_sqd and stores in results.
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: Vector of length L of squared bandwidths for the kernels  
        exp(-||x-y||_2^2/lam_sqd[l])
      results: Zeros vector of length L in which results will be stored
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j, k, l
    cdef double arg
    for i in range(n1):
        for j in range(n2):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X2[j,k])**2
            for l in range(len(lam_sqd)):
                results[l] += exp(-arg/lam_sqd[l])       
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_gaussian_kernel_same_aggregated(const double[:,:] X1,
                                               const double[:] lam_sqd,
                                               double[:] results) noexcept nogil:
    """
    Computes the sum of Gaussian kernel evaluations between all rows of X1 and all rows of X1,
    for all the squared bandwidths in lam_sqd and stores in results.
    
    Args:
      X1: Matrix of size (n1,d)
      lam_sqd: Vector of length L of squared bandwidths for the kernels 
        exp(-||x-y||_2^2/lam_sqd[l])
      results: Zeros vector of length L in which results will be stored
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j, k, l
    cdef double arg
    for i in range(n1):
        for j in range(i+1):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X1[j,k])**2
            if j < i:
                for l in range(len(lam_sqd)):
                    results[l] += 2*exp(-arg/lam_sqd[l])
            else:
                for l in range(len(lam_sqd)):
                    results[l] += exp(-arg/lam_sqd[l])

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double biased_sqMMD_gaussian(const double[:,:] X1,
                                   const double[:,:] X2,
                                   const double lam_sqd) noexcept nogil:
    """
    Computes the biased quadratic squared MMD estimator for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n12,d)
      lam: double, bandwidth of the kernel
    """
    
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef double first_term = sum_gaussian_kernel(X1,X1,lam_sqd)
    cdef double second_term = sum_gaussian_kernel(X1,X2,lam_sqd)
    cdef double third_term = sum_gaussian_kernel(X2,X2,lam_sqd)
    cdef double bsqMMD = first_term/(n1*n1) - 2*second_term/(n1*n2) + third_term/(n2*n2)
            
    return(bsqMMD)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double unbiased_sqMMD_gaussian(const double[:,:] X1,
                                     const double[:,:] X2,
                                     const double lam_sqd) noexcept nogil:
    """
    Computes the unbiased quadratic squared MMD estimator for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
    """
    
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef double first_term = sum_gaussian_kernel_same(X1,lam_sqd)
    cdef double second_term = sum_gaussian_kernel(X1,X2,lam_sqd)
    cdef double third_term = sum_gaussian_kernel_same(X2,lam_sqd)
    cdef double extra_terms_first = sum_gaussian_kernel_linear_eval(X1,X1,lam_sqd)
    cdef double extra_terms_third = sum_gaussian_kernel_linear_eval(X2,X2,lam_sqd)
    cdef double usqMMD = (first_term-extra_terms_first)/(n1*(n1-1)) - 2*second_term/(n1*n2) + (third_term-extra_terms_third)/(n2*(n2-1))
            
    return(usqMMD)
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void block_sqMMD_gaussian(const double[:,:] X1,
                                const double[:,:] X2,
                                const double lam_sqd,
                                const long block_size,
                                double[:] results) noexcept nogil:
    """Computes the block quadratic squared MMD estimator for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n12,d)
      lam_sqd: double, squared bandwidth of the kernel
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
                h_value = h_gaussian(X1[j],X1[k],X2[j],X2[k],lam_sqd)
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
cpdef void block_sqMMD_gaussian_reordered(const double[:,:] X1,
                                          const double[:,:] X2,
                                          const double lam_sqd,
                                          const long block_size,
                                          const long seed,
                                          long[:] epsilon,
                                          double[:] results) noexcept nogil:
    """Computes the block squared MMD estimator for the Gaussian kernel from samples X1 and X2, reordering before constructing blocks
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n12,d)
      lam_sqd: double, squared bandwidth of the kernel
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
                h_value = h_gaussian(X1[j],X1[k],X2[j],X2[k],lam_sqd)
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
cpdef double block_sqMMD_gaussian_Rademacher(const double[:,:] X1,
                                             const double[:,:] X2,
                                             const double lam_sqd,
                                             const long block_size,
                                             const long long[:,:] epsilon,
                                             double[:] split_values,
                                             double[:] sqMMD_values) noexcept nogil:
    """Computes the block squared MMD estimator for the Gaussian kernel from samples X1 and X2, 
    for different Rademacher vectors
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, bandwidth of the kernel
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
                h_value = h_gaussian(X1[j],X1[k],X2[j],X2[k],lam_sqd)
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
cpdef double h_gaussian(const double[:] x1,
                        const double[:] x2,
                        const double[:] y1,
                        const double[:] y2,
                        const double lam_sqd) noexcept nogil:
    """
    Computes the kernel function h((x1,y1),(x2,y2)) = k(x1,x1) + k(y2,y2) - k(x1,y2) - k(x2,y1), where k is the Gaussian kernel
    
    Args:
      x1: vector of size d
      x2: vector of size d
      y1: vector of size d
      y2: vector of size d
      lam_sqd: squared bandwidth
    """
    
    cdef double arg_x1_x2 = 0
    cdef double arg_y1_y2 = 0
    cdef double arg_x1_y2 = 0
    cdef double arg_x2_y1 = 0
    cdef long k 
    for k in range(len(x1)):
        arg_x1_x2 += (x1[k]-x2[k])**2
        arg_y1_y2 += (y1[k]-y2[k])**2
        arg_x1_y2 += (x1[k]-y2[k])**2
        arg_x2_y1 += (x2[k]-y1[k])**2
    return(exp(-arg_x1_x2/lam_sqd) + exp(-arg_y1_y2/lam_sqd) - exp(-arg_x1_y2/lam_sqd) - exp(-arg_x2_y1/lam_sqd))

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void h_gaussian_aggregated(const double[:] x1,
                                 const double[:] x2,
                                 const double[:] y1,
                                 const double[:] y2,
                                 const double[:] lam_sqd,
                                 double[:] h_values) noexcept nogil:
    """
    Computes the kernel function h((x1,y1),(x2,y2)) = k(x1,x1) + k(y2,y2) - k(x1,y2) - k(x2,y1), where k is the Gaussian kernel,
    for all the squared bandwidths in lam_sqd
    
    Args:
      x1: vector of size d
      x2: vector of size d
      y1: vector of size d
      y2: vector of size d
      lam_sqd: squared bandwidth
      h_values: used to store results
    """
    
    cdef double arg_x1_x2 = 0
    cdef double arg_y1_y2 = 0
    cdef double arg_x1_y2 = 0
    cdef double arg_x2_y1 = 0
    cdef long k 
    for k in range(len(x1)):
        arg_x1_x2 += (x1[k]-x2[k])**2
        arg_y1_y2 += (y1[k]-y2[k])**2
        arg_x1_y2 += (x1[k]-y2[k])**2
        arg_x2_y1 += (x2[k]-y1[k])**2
    for k in range(len(lam_sqd)):
        h_values[k] = exp(-arg_x1_x2/lam_sqd[k]) + exp(-arg_y1_y2/lam_sqd[k]) - exp(-arg_x1_y2/lam_sqd[k]) - exp(-arg_x2_y1/lam_sqd[k])

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void incomplete_sqMMD_gaussian(const double[:,:] X1,
                                     const double[:,:] X2,
                                     const long long[:] incomplete_list,
                                     const long long max_l,
                                     const double lam_sqd,
                                     const long seed,
                                     double[:] sqMMD_list,
                                     double[:] sqMMD_variances,
                                     double[:] sqMMD_times) noexcept nogil:
    """
    Computes the incomplete squared MMD estimator
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      incomplete_list: vector containing numbers of pairs, ordered
      max_l: largest number of pairs (largest/last element of incomplete_list)
      lam_sqd: squared bandwidth
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
        h_value = h_gaussian(X1[D0],X1[D1],X2[D0],X2[D1],lam_sqd)
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
cpdef void incomplete_sqMMD_gaussian_Rademacher_subdiagonals(const double[:,:] X1,
                                                             const double[:,:] X2,
                                                             const long long[:] incomplete_list,
                                                             const long B,
                                                             const long long max_l,
                                                             const double lam_sqd,
                                                             const long[:,:] epsilon,
                                                             double[:,:] sqMMD_matrix,
                                                             double[:] sqMMD_vector,
                                                             double[:] sqMMD_times) noexcept nogil:
    """
    Computes the incomplete squared MMD estimator for the Gaussian kernel from samples X1 and X2, 
    for different Rademacher vectors
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      incomplete_list: vector containing numbers of pairs, ordered
      B: number of Rademacher vectors
      max_l: largest number of pairs (largest/last element of incomplete_list)
      lam_sqd: squared bandwidth
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
    
    printf('Run ktc.incomplete_sqMMD_gaussian_Rademacher inside\n')
    
    D0 = 0
    D1 = 1
    r = 1
    for j in range(max_l):
        if r > n_samples - 1:
            printf('Error: r > n_samples - 1: r = %u, n_samples = %u, D0 = %u, D1 = %u \n', r, n_samples, D0, D1)
        
        #Compute contribution of pair
        h_value = h_gaussian(X1[D0],X1[D1],X2[D0],X2[D1],lam_sqd)
        
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
cpdef void incomplete_sqMMD_gaussian_Rademacher_aggregated(const double[:,:] X1,
                                                           const double[:,:] X2,
                                                           const long long[:] incomplete_list,
                                                           const long B,
                                                           const long long max_l,
                                                           const double[:] lam_sqd,
                                                           const long[:,:] epsilon,
                                                           double[:,:,:] sqMMD_matrix,
                                                           double[:,:] sqMMD_vector,
                                                           double[:] sqMMD_times,
                                                           double[:] h_values) noexcept nogil:
    """
    Computes the incomplete squared MMD estimator for the Gaussian kernel from samples X1 and X2, 
    for different Rademacher vectors and different bandwidths
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      incomplete_list: vector containing numbers of pairs, ordered
      max_l: largest number of pairs (largest/last element of incomplete_list)
      epsilon: used to store Rademacher vector
      lam_sqd: squared bandwidth
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
    
    printf('Run ktc.incomplete_sqMMD_gaussian_Rademacher inside\n')
    
    D0 = 0
    D1 = 1
    r = 1
    for j in range(max_l):
        if r > n_samples - 1:
            printf('Error: r > n_samples - 1: r = %u, n_samples = %u, D0 = %u, D1 = %u \n', r, n_samples, D0, D1)
        
        #Compute contribution of pair
        h_gaussian_aggregated(X1[D0],X1[D1],X2[D0],X2[D1],lam_sqd,h_values)
        
        #Update sqMMD vector
        for i in range(len(lam_sqd)):
            for l in range(B):
                sqMMD_vector[i,l] += epsilon[D0,l]*epsilon[D1,l]*h_values[i]/n_samples
            sqMMD_vector[i,B] += h_values[i]/n_samples
        
        #If j is equal to some element of incomplete_list (minus 1), store sqMMD values and times    
        if j == incomplete_list[incomplete_index] - 1:
            printf('incomplete_index %u of %u: %u\n', incomplete_index+1, len(incomplete_list), incomplete_list[incomplete_index])
            for i in range(len(lam_sqd)):
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
        for i in range(len(lam_sqd)):
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
                             const double lam_sqd) noexcept nogil:
    """
    Computes E_z[(E_{z'}h(z,z'))^2]
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double total_sum = 0
    cdef double intermediate_sum
    for i in range(n_samples):
        intermediate_sum = 0
        for j in range(n_samples):
            intermediate_sum += h_gaussian(X1[i],X1[j],X2[i],X2[j],lam_sqd)
        total_sum += (intermediate_sum/n_samples)**2
    return(total_sum/n_samples)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double expectation_h_2(const double[:,:] X1,
                             const double[:,:] X2,
                             const double lam_sqd) noexcept nogil:
    """
    Computes E_{z,z'}[(h(z,z'))^2]
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double total_sum = 0
    for i in range(n_samples):
        for j in range(n_samples):
            total_sum += (h_gaussian(X1[i],X1[j],X2[i],X2[j],lam_sqd))**2
    return(total_sum/n_samples**2)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double expectation_h_3(const double[:,:] X1,
                             const double[:,:] X2,
                             const double lam_sqd) noexcept nogil:
    """
    Compute (E_{z,z'}[h(z,z')])^2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double total_sum = 0
    for i in range(n_samples):
        for j in range(n_samples):
            total_sum += h_gaussian(X1[i],X1[j],X2[i],X2[j],lam_sqd)
    return (total_sum/n_samples**2)**2

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sigma_1_sqd(const double[:,:] X1,
                         const double[:,:] X2,
                         const double lam_sqd) noexcept nogil:
    """
    Compute sigma_1^2 = E_z[(E_{z'}h(z,z'))^2] - (E_{z,z'}[h(z,z')])^2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
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
            h_value = h_gaussian(X1[i],X1[j],X2[i],X2[j],lam_sqd)
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
                         const double lam_sqd) noexcept nogil:
    """
    Compute sigma_2^2 = E_{z,z'}[h(z,z')^2] - E_z[(E_{z'}h(z,z'))^2]
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
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
            h_value = h_gaussian(X1[i],X1[j],X2[i],X2[j],lam_sqd)
            intermediate_sum += h_value
            first_term += h_value**2
        second_term += (intermediate_sum/n_samples)**2
    return first_term/n_samples**2 - (second_term/n_samples)
