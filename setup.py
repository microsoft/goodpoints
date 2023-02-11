from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from os.path import join, dirname
import numpy

# Paths to numpy libraries
lib_path_npyrandom = join(numpy.get_include(), '..', '..', 'random', 'lib')
lib_path_npymath = join(numpy.get_include(), '..', 'lib')

# Cython modules to be compiled
extensions = [
    Extension(
        "goodpoints.ktc", [join("goodpoints","ktc.pyx")],
        extra_compile_args=['-O3'],
        language="c", include_dirs=[numpy.get_include()],
        library_dirs=[lib_path_npyrandom,lib_path_npymath],
        libraries=['npyrandom','npymath','m']
    ),
    Extension(
        "goodpoints.compressc", [join("goodpoints","compressc.pyx")],
        extra_compile_args=['-O3'],
        language="c", include_dirs=[numpy.get_include()],
        library_dirs=[lib_path_npyrandom,lib_path_npymath],
        libraries=['npyrandom','npymath','m'],
    ), 
    Extension(
        "goodpoints.cttc", [join("goodpoints","cttc.pyx")],
        extra_compile_args=['-O3'],
        language="c", include_dirs=[numpy.get_include()],
        library_dirs=[lib_path_npyrandom,lib_path_npymath],
        libraries=['npyrandom','npymath','m'],
    ), 
    Extension(
        "goodpoints.gaussianc", [join("goodpoints","gaussianc.pyx")],
        extra_compile_args=['-O3'],
        language="c", include_dirs=[numpy.get_include()],
        library_dirs=[lib_path_npyrandom,lib_path_npymath],
        libraries=['npyrandom','npymath','m'],
    ), 
]

# Path to Cython declaration (PXD) files 
include_path = ["goodpoints"]

setup(ext_modules=cythonize(extensions, language_level = "3",
                            include_path=include_path))
