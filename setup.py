import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

ext_modules = [Extension("pybm3d.bm3d",
                         sources=["pybm3d/bm3d.pyx",
                                  "bm3d_src/mt19937ar.c",
                                  "bm3d_src/io_png.c",
                                  "bm3d_src/bm3d.cpp",
                                  "bm3d_src/lib_transforms.cpp",
                                  "bm3d_src/utilities.cpp",
                         ],
                         language="c++",
                         libraries=["png", "fftw3", "fftw3f"])]

setup(
    name='pybm3d',
    version='0.1.0',
    description='Python wrapper around BM3D',
    author='Eric Jonas',
    author_email='jonas@ericjonas.com',
    url='https://github.com/ericmjonas/pybm3d',
    packages=['pybm3d'],
    ext_modules=ext_modules,
    cmdclass={'build_ext':build_ext},
    setup_requires=['numpy'],
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27',
        'numpy>=1.13',
    ],
    tests_require=[
        'pytest',
        'scikit-image',
    ],

)
