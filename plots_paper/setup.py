from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

ext_modules=[
    Extension("arcs",        ["arcs.pyx"],        include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),
    Extension("angles",      ["angles.pyx"],      include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),
    Extension("lenghts",     ["lenghts.pyx"],     include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),    
    Extension("widths",      ["widths.pyx"],      include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),
    Extension("LWcross",     ["LWcross.pyx"],     include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),
]

setup(
  name = 'Elliptical Sources',
  cmdclass = {'build_ext': build_ext},
  ext_modules = cythonize(ext_modules),
)
