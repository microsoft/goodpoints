"""Compress Then Test and Random Fourier Features functionality

Functionality supporting the CTT, ACTT, and LR-CTT-RFF tests of
  Carles Domingo-Enrich, Raaz Dwivedi, and Lester Mackey.
  Compress Then Test: Powerful Kernel Testing in Near-linear Time.
  https://arxiv.org/pdf/2301.05974.pdf
  
and the RFF test (with permutation-based null) of
  Ji Zhao and Deyu Meng.
  FastMMD: Ensemble of Circular Discrepancy for Efficient Two-Sample Test.
  https://arxiv.org/pdf/1405.2664.pdf
"""
import numpy as np
cimport numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport cython
from libc.math cimport sqrt, cos
# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. 
np.import_array()

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CTT Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double signed_matrix_sum(const double[:,:] K,
                               const long [:] signs) noexcept nogil:
    """
    Returns sum_{i,j} K[i,j] signs[i] signs[j]
    
    Args:
      K: symmetric matrix of size (n,n) (only lower triangular part is used)
      signs: vector of length n containing +/-1 values
    """
    
    cdef long n = K.shape[0]
    cdef long i, j
    cdef double val = 0
    cdef double sign_i
    
    for i in range(n):
        sign_i = signs[i]
        val += K[i,i]
        for j in range(i):
            # Count K[i,j] twice, since K[i,j] = K[j,i]
            val += 2 * K[i,j] * sign_i * signs[j]
    return val

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ACTT Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void signed_tensor_sum(const double[:,:,:] K,
                             const long [:] signs,
                             double[:] K_sum) noexcept nogil:
    """
    Computes sum_{i,j} K[i,j,:] signs[i] signs[j] and stores in K_sum
    
    Args:
      K: tensor of size (n,n,L); K[:,:,l] represents a symmetric matrix for 
        each l; only lower triangular part of that matrix is used by this code
      signs: vector of length n containing +/-1 values
      K_sum: vector of zeros of length l for storing the results
    """
    
    cdef long n = K.shape[0]
    cdef long L = K.shape[2]
    cdef long i, j, l
    cdef double sign_i, two_sign_ij
    
    for i in range(n):
        sign_i = signs[i]
        for j in range(i):
            # Count K[i,j,:] twice, since K[i,j,:] = K[j,i,:]
            two_sign_ij = 2 * sign_i * signs[j]
            for l in range(L):
                K_sum[l] += K[i,j,l] * two_sign_ij
        
        for l in range(L):
            K_sum[l] += K[i,i,l]

            
'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CTT-RFF Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void linear_kernel_same(const double[:,:] X1,
                              double[:,:] K) noexcept nogil:
    """
    Stores the linear kernel matrix X1 * X1^T in K
    
    Args:
      X1: Matrix of size (n1,d)
      K: Empty matrix of size (n1,n1) to store kernel matrix
    """
    cdef long n1 = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j, k
    cdef double arg
    
    for i in range(n1):
        for j in range(i+1):
            arg = 0
            for k in range(d):
                arg += X1[i,k]*X1[j,k]
            K[i,j] = arg
            K[j,i] = K[i,j] 


'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Random Fourier Features Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''   
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void rff(const double[:,:] X1,
               const double[:,:] X2,
               const double[:,:] W,
               const double[:] b,
               double[:,:] features) noexcept nogil:
    """
    Computes the features cos(W x + b)/sqrt(n_features) for each row x 
    of [X1; X2], averages those feature vectors across consecutive bins of 
    size bin_size, and stores the results in features
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      W: Matrix of size (n_features,d) 
      b: Vector of size (n_features)
      features: Zeros matrix of size ((n1+n2)/bin_size, n_features) used to 
        store the averaged features
    """
    
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long num_bins = features.shape[0]
    cdef long bin_size = (n1+n2) // num_bins
    cdef long num_bins1 = n1 // bin_size
    cdef long num_bins2 = n2 // bin_size
    cdef long d = X1.shape[1]
    cdef long n_features = W.shape[0]
    
    cdef long i, j, k, l, i_bin
    cdef double arg
    
    # Add X1 features to appropriate bin
    # Keep track of index i into X1 as we iterate over each bin
    i = 0
    for i_bin in range(num_bins1):
        for l in range(bin_size):
            for j in range(n_features):
                arg = 0
                for k in range(d):
                    arg += X1[i,k]*W[j,k]
                features[i_bin,j] += cos(arg + b[j])
            i += 1
            
    # Add X2 features to appropriate bin
    # Keep track of index i into X2 as we iterate over each bin
    i = 0
    for i_bin in range(num_bins2):
        for l in range(bin_size):
            for j in range(n_features):
                arg = 0
                for k in range(d):
                    arg += X2[i,k]*W[j,k]
                features[num_bins1+i_bin,j] += cos(arg + b[j])
            i += 1
            
    # Multiply all features by sqrt(2), divide all features by bin_size 
    # to replace sum with average and normalize across features by dividing 
    # by sqrt(n_features)
    cdef double normalizer = 1./(bin_size*sqrt(n_features/2.))
    for k in range(num_bins):
        for j in range(n_features):
            features[k,j] *= normalizer

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double signed_vector_sqd_norm(const double[:,:] X,
                                    const long [:] signs) noexcept nogil:
    """
    Returns sum_j (sum_{i} X[i,j] signs[i])^2
    
    Args:
      X: matrix of size (n,d)
      signs: vector of length n containing +/-1 values
    """
    
    cdef long n = X.shape[0]
    cdef long d = X.shape[1]
    cdef long i, j
    cdef double result = 0
    cdef double term
    
    for j in range(d):
        term = 0
        for i in range(n):
            term += X[i,j] * signs[i]
        result += term**2

    return result
