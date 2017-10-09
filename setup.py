"""PyBM3D packaging and distribution."""
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class CustomBuildExt(_build_ext):
    """Custom build extension class."""

    def __init__(self, *args, **kwargs):
        """Extends default class constructor."""
        # _build_ext is not a new-style (object) class, which is required for
        # super to work
        _build_ext.__init__(self, *args, **kwargs)
        self.cuda_config = self.load_cuda_config()

    def build_extensions(self):
        """Extends default build_extensions method.

        Further customizes the compiler to add C file specific compile
        arguments and support nvcc compilation of *.cu CUDA files."""

        self.customize_compiler_for_c_args_and_nvcc()
        _build_ext.build_extensions(self)

    def finalize_options(self):
        """Extends default finalize_options method.

        Injects NumPy`s C include directories into the Cython compilation. This
        is done after NumPy was installed through the setuptools setup_requires
        argument which removes NumPy from the necessary preinstalled
        packages."""

        _build_ext.finalize_options(self)
        # prevent numpy from thinking it is still in its setup process
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

    def customize_compiler_for_c_args_and_nvcc(self):
        """Customize the compiler.

        The customization adds C file specific compile
        arguments and support for nvcc compilation of *.cu CUDA files."""

        self.compiler.src_extensions.append('.cu')

        # save references to the default compiler_so and _comple methods
        default_compiler_so = self.compiler.compiler_so
        super = self.compiler._compile

        # now redefine the _compile method. This gets executed for each
        # object but distutils doesn't have the ability to change compilers
        # based on source extension: we add it.
        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if os.path.splitext(src)[1] == '.cu':
                # use the nvcc for *.cu files
                self.compiler.set_executable('compiler_so',
                                             self.cuda_config['nvcc'])
                postargs = extra_postargs['nvcc']
            else:
                postargs = extra_postargs['unix']

                # add C file specific compile arguments
                if os.path.splitext(src)[1] == '.c':
                    postargs = postargs + extra_postargs['c_args']

            super(obj, src, ext, cc_args, postargs, pp_opts)
            # reset the default compiler_so
            self.compiler.compiler_so = default_compiler_so

        # inject our redefined _compile method into the class
        self.compiler._compile = _compile

    @staticmethod
    def load_cuda_config():
        """Locate the CUDA environment on the system
        Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
        and values giving the absolute path to each directory.
        """
        def find_in_path(name, path):
            """Finds a file by name in a search path."""
            for dir in path.split(os.pathsep):
                binpath = os.path.join(dir, name)
                if os.path.exists(binpath):
                    return os.path.abspath(binpath)
            return None

        # first check if the CUDA_HOME env variable is in use
        if 'CUDA_HOME' in os.environ:
            home = os.environ['CUDA_HOME']
            nvcc = os.path.join(home, 'bin', 'nvcc')
        else:
            # otherwise, search the PATH for NVCC
            nvcc = find_in_path('nvcc', os.environ['PATH'])
            if nvcc is None:
                return {'cuda_available': False}
                raise EnvironmentError('The nvcc binary could not be located '
                                       'in your $PATH. Either add it to your '
                                       'path, or set $CUDA_HOME')
            home = os.path.dirname(os.path.dirname(nvcc))

        cuda_config = {'home': home, 'nvcc': nvcc,
                       'include': os.path.join(home, 'include'),
                       'lib64': os.path.join(home, 'lib64')}
        for k, v in cuda_config.items():
            if not os.path.exists(v):
                raise EnvironmentError('The CUDA %s path could not be located '
                                       'in %s' % (k, v))

        return cuda_config


# optimize to the current CPU and enable warnings
extra_compile_args = {'unix': ['-march=native', '-Wall', '-Wextra', ],
                      'c_args': ['-std=c99', ]}
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
    version='0.2.1',
    description='Python wrapper around BM3D',
    author='Eric Jonas',
    author_email='jonas@ericjonas.com',
    maintainer='Tim Meinhardt',
    maintainer_email='meinhardt.tim@gmail.com',
    url='https://github.com/ericmjonas/pybm3d',
    download_url='https://github.com/ericmjonas/pybm3d/releases/download/v0.2.1/pybm3d-0.2.1.tar.gz',
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
