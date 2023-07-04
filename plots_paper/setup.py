from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("arcs.pyx")) 

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

ext_modules=[
    Extension("arcs",        ["arcs.pyx"],        include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),
    Extension("angles",      ["angles.pyx"],      include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),
    Extension("lengths",     ["lengths.pyx"],     include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),    
    Extension("widths",      ["widths.pyx"],      include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),
    Extension("LWcross",     ["LWcross.pyx"],     include_dirs=[numpy.get_include()], extra_compile_args=["-O3"]),
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"} # all are Python-3
    
setup(
  name = 'Elliptical Sources',
  cmdclass = {'build_ext': build_ext},
  ext_modules = cythonize(ext_modules),
)
