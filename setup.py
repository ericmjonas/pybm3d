"""PyBM3D packaging and distribution."""
# pylint: disable=invalid-name
import sys
import os
import re
import tempfile
from textwrap import dedent

import distutils.sysconfig  # pylint: disable=import-error
import distutils.ccompiler  # pylint: disable=import-error, no-name-in-module
from distutils.errors import CompileError, LinkError  # pylint: disable=import-error, no-name-in-module

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
        default_compile = self.compiler._compile  # pylint: disable=protected-access

        # now redefine the _compile method. This gets executed for each
        # object but distutils doesn't have the ability to change compilers
        # based on source extension: we add it.
        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # pylint: disable=too-many-arguments
            if os.path.splitext(src)[1] == '.cu':
                self.compiler.set_executable('compiler_so',
                                             self.cuda_config['nvcc'])
                postargs = extra_postargs['nvcc']
            else:
                postargs = extra_postargs['unix']

                # add C file specific compile arguments
                if os.path.splitext(src)[1] == '.c':
                    postargs = postargs + extra_postargs['c_args']

            default_compile(obj, src, ext, cc_args, postargs, pp_opts)
            # reset the default compiler_so
            self.compiler.compiler_so = default_compiler_so

        # inject our redefined _compile method into the class
        self.compiler._compile = _compile  # pylint: disable=protected-access

    @staticmethod
    def load_cuda_config():
        """Locate the CUDA environment on the system.

        Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
        and values giving the absolute path to each directory."""

        def find_in_path(name, path):
            """Finds a file by name in a search path."""
            for directory in path.split(os.pathsep):
                binpath = os.path.join(directory, name)
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
                # raise EnvironmentError('The nvcc binary could not be located '
                #                        'in your $PATH. Either add it to your '
                #                        'path, or set $CUDA_HOME')
            home = os.path.dirname(os.path.dirname(nvcc))

        cuda_config = {'home': home, 'nvcc': nvcc,
                       'include': os.path.join(home, 'include'),
                       'lib64': os.path.join(home, 'lib64')}
        for k, v in cuda_config.items():
            if not os.path.exists(v):
                raise EnvironmentError('The CUDA %s path could not be located '
                                       'in %s' % (k, v))

        return cuda_config


def find_library(library, header_file=None, extra_preargs=None):
    """Find C library.

    Naively tries to compile some C code with the respective library and
    assumes that the library is not available if compilation fails.
    """
    if header_file is None:
        header_file = library + '.h'
    if extra_preargs is None:
        extra_preargs = []

    compiler = distutils.ccompiler.new_compiler()  # pylint: disable=no-member
    distutils.sysconfig.customize_compiler(compiler)

    c_code = dedent("""
    #include <{}>

    int main(int argc, char* argv[])
    {{
        return 0;
    }}
    """.format(header_file))

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.c') as tmp_file:
        tmp_file.write(c_code)
        tmp_file.flush()

        try:
            link_objects = compiler.compile([tmp_file.name],
                                            extra_preargs=extra_preargs,
                                            output_dir="/")
            compiler.link_executable(link_objects,
                                     # output_progname
                                     os.path.splitext(tmp_file.name)[0],
                                     libraries=[library],)
        except (CompileError, LinkError):
            print('The {} library was not found.'.format(library))
            return False
        else:
            return True


if not find_library('fftw3'):
    raise ImportError('\nFFTW3 C library is not installed. Please install:\n'
                      '\tLinux: `sudo apt-get install libfftw3-dev`\n'
                      '\tOSX: `brew update && brew install fftw`')

# optimize to the current CPU and enable warnings
extra_compile_args = {'unix': ['-march=native', '-Wall', '-Wextra', ],
                      'c_args': ['-std=c99', ]}
libraries = ['png', 'tiff', 'jpeg', 'fftw3', 'fftw3f', ]

#
# OpenMP
#

# Default OSX compiler clang does not support OpenMP. Therefore, do not use
# clang if other compilers are available.
if 'darwin' in sys.platform:
    file_list = os.listdir("/usr/local/bin")
    regex = re.compile(r'^gcc-[1-9]')
    sort_compiler_file_list = sorted(list(filter(regex.search, file_list)))
    if sort_compiler_file_list:
        # get compiler with highest version number
        compiler_file = sort_compiler_file_list[-1]
        os.environ["CC"] = compiler_file
        os.environ["CXX"] = compiler_file.replace('c', '+')

# activate OpenMP if library and compiler support are available
if find_library('gomp', 'omp.h', ['-fopenmp']):
    extra_compile_args['unix'] += ['-fopenmp']
    libraries += ['gomp']

ext_modules = [Extension("pybm3d.bm3d",
                         language="c++",
                         sources=["pybm3d/bm3d.pyx",
                                  "bm3d_src/iio.c",
                                  "bm3d_src/bm3d.cpp",
                                  "bm3d_src/lib_transforms.cpp",
                                  "bm3d_src/utilities.cpp", ],
                         extra_compile_args=extra_compile_args,
                         libraries=libraries)]

# subprocess.check_output(['git', 'tag']).split()[-1][1:].decode("utf-8")
version = '0.2.1'
url = 'https://github.com/ericmjonas/pybm3d'
download_url = os.path.join(url,
                            'releases/download/v' + version,
                            'pybm3d-' + version + '.tar.gz')
setup(name='pybm3d',
      version=version,
      description='BM3D denoising for Python',
      author='Eric Jonas',
      author_email='jonas@ericjonas.com',
      maintainer='Tim Meinhardt',
      maintainer_email='meinhardt.tim@gmail.com',
      url=url,
      download_url=download_url,
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
          'scikit-image', ],)
