from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = []
ext_modules += cythonize([Extension("pybm3d.bm3d", ["pybm3d/bm3d.pyx"])])


setup(
    name='pybm3d',
    ext_modules=ext_modules,

)
