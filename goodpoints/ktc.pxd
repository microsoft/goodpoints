"""Cython declarations for kernel thinning functionality used by other files
"""
from numpy.random cimport bitgen_t

cdef void thin_K(const double[:, :] K,
                 bitgen_t* rng,
                 const double delta,
                 const bint unique,
                 const bint mean0,
                 double[:] aux_double_mem,
                 long[:,:] aux_long_mem,
                 long[:] output_indices) noexcept nogil