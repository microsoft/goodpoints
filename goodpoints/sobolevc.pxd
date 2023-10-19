"""Cython declarations for Sobolev kernel functionality used by other files
"""
cdef double sobolev_kernel_two_points(const double[:] X1,
                                      const double[:] X2,
                                      const double[:] s) noexcept nogil

cdef double sobolev_kernel_one_point(const double[:] X1,
                                     const double[:] s) noexcept nogil