"""PyBM3D packaging and distribution."""
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        if os.path.splitext(src)[1] == '.c':
            postargs.append('-std=c99')

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


class CustomBuildExt(_build_ext):
    """Custom build extension class.

    The class injects NumPy C include directories into the Cython compilation
    after NumPy was installed and is available. This allows us to not depend on
    a preinstalled NumPy but automatically install NumPy with the
    setup_requires arguement.
    """

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        _build_ext.build_extensions(self)

    def finalize_options(self):
        """Injects Numpy C include directories.

        Overrides default finalize_options method.
        """
        _build_ext.finalize_options(self)
        # prevent numpy from thinking it is still in its setup process
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


# optimize to the current CPU and enable warnings
extra_compile_args = {'gcc': ['-march=native', '-Wall', '-Wextra', ],
                      'clang': ['-march=native', '-Wall', '-Wextra', ]}
libraries = ["png", "tiff", "jpeg", "fftw3", "fftw3f"]
ext_modules = [Extension("pybm3d.bm3d",
                         language="c++",
                         sources=["pybm3d/bm3d.pyx",
                                  "bm3d_src/iio.c",
                                  "bm3d_src/bm3d.cpp",
                                  "bm3d_src/lib_transforms.cpp",
                                  "bm3d_src/utilities.cpp", ],
                         extra_compile_args=extra_compile_args,
                         libraries=libraries)]

setup(
    name='pybm3d',
    version='0.1.0',
    description='Python wrapper around BM3D',
    author='Eric Jonas',
    author_email='jonas@ericjonas.com',
    url='https://github.com/ericmjonas/pybm3d',
    zip_safe=False,
    packages=['pybm3d'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': CustomBuildExt},
    setup_requires=['numpy>=1.13', ],
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27',
        'numpy>=1.13', ],
    tests_require=[
        'pytest',
        'scikit-image', ],
)
