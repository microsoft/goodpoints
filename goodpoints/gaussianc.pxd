"""Cython declarations for Gaussian kernel functionality used by other files
"""
cdef double gaussian_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const double[:] lam_sqd) nogil