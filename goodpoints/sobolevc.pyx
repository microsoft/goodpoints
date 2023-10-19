"""Sobolev kernel functionality.

Cython implementation of functions involving Sobolev kernel evaluation.
"""
import numpy as np
cimport numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport cython
from libc.math cimport pi
# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. 
np.import_array()

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Sobolev Kernel Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

cdef double PI_SQD = pi * pi
cdef double PI_TO_4 = PI_SQD * PI_SQD
cdef double PI_TO_6 = PI_TO_4 * PI_SQD

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double single_sobolev_kernel_two_points(const double[:] X1,
                                             const double[:] X2,
                                             const double s) noexcept nogil:
    """
    Computes a single Sobolev kernel 
    k(X1, X2, s) = -1 + prod_j (1 + [ (-1)^{s-1}(2pi)^{2s} / (2s)! ] B_{2s}({X1[j]-X2[j]}))
    between two points X1 and X2, where B_{2s} is the order 2s Bernoulli polynomial
    and {x} = x - floor(x) = x + indic(x < 0) represents the fractional part of x.

    Note: Subtracting 1 in the definition of k ensures E[k(X, y, s)] = 0 
    when X ~ Unif(0,1)
    
    Args:
      X1: array of size d
      X2: array of size d
      s: sobolev kernel smoothness; valid options are 1., 2., 3.
    """
    cdef long d = X1.shape[0]
    cdef double ans, x, x_sqd, x_to_4
    cdef long j
    
    ans = 1.
    if s == 1:
        for j in range(d):
            # B_{2}(z) = z^2 - z + 1./6
            x = X1[j]-X2[j]
            if x <0:
                x += 1.
            ans *= (1. + 2. * PI_SQD * (x * x - x + 1. / 6.))
    elif s == 2:
        for j in range(d):
            # B_{4}(z) = z^4 - 2z^3 + z^2 - 1./30
            x = X1[j]-X2[j]
            if x <0:
                x += 1.
            x_sqd = x * x
            ans *= ( 1. - PI_TO_4 * 2. / 3. * \
                (x_sqd * x_sqd - 2. * (x_sqd * x) + x_sqd - 1. / 30.))
    else: # s == 3:
        for j in range(d):
            # B_{6}(z) = z^6 - 3z^5 + 5 z^4 / 2 - z^2 / 2 + 1./42
            x = X1[j]-X2[j]
            if x <0:
                x += 1.
            x_sqd = x * x
            x_to_4 = x_sqd * x_sqd
            ans *= ( 1. + PI_TO_6 * 4. / 45. * ((x_sqd*x_to_4) - 3 * (x*x_to_4) + \
                    5. / 2. * x_to_4 - x_sqd / 2. + 1. / 42.))
    return(ans-1.)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double sobolev_kernel_two_points(const double[:] X1,
                                      const double[:] X2,
                                      const double[:] s) noexcept nogil:
    """
    Computes a sum of Sobolev kernels sum_j k(X1, X2, s_j) 
    between two points X1 and X2
    
    Args:
      X1: array of size d
      X2: array of size d
      s: array of sobolev kernel smoothnessnesses 
        (see single_sobolev_kernel_two_points)
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

# Diagonal values for Sobolev kernels when d = 1
cdef double sobolev_base_1 = (1. + 2. * PI_SQD / 6.)
cdef double sobolev_base_2 = (1. + PI_TO_4 * 2. / 90.)
cdef double sobolev_base_3 = (1 + PI_TO_6 * 2. / 945.)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double single_sobolev_kernel_one_point(const double[:] X1,
                                            const double s) noexcept nogil:
    """
    Computes a single Sobolev kernel k(X1, X1, s) 
    between X1 and itself
    
    Args:
      X1: array of size d
      s: sobolev kernel smoothness; valid options are 1., 2., 3.
    """
    cdef long d = X1.shape[0]
    cdef double ans, base

    if s == 1:
        base = sobolev_base_1
    elif s == 2:
        base = sobolev_base_2
    else: # s == 3:
        base = sobolev_base_3
    ans = base
    for j in range(1, d):
        ans *= base
    return(ans-1)


@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double sobolev_kernel_one_point(const double[:] X1,
                                     const double[:] s) noexcept nogil:
    """
    Computes a sum of Sobolev kernels sum_j k(X1, X1, s_j) 
    between X1 and itself
    
    Args:
      X1: array of size d
      s: array of sobolev kernel smoothnessnesses 
        (see single_sobolev_kernel_two_points)
    """
    
    cdef long num_kernels = s.shape[0]
    
    cdef long j
    cdef double kernel_sum
    
    # Compute the kernel sum
    kernel_sum = single_sobolev_kernel_one_point(X1, s[0])
    for j in range(1, num_kernels):
        kernel_sum += single_sobolev_kernel_one_point(X1, s[j])
    return(kernel_sum)

