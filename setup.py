"""PyBM3D packaging and distribution."""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class BuildExt(_build_ext):
    """Custom build extension class.

    The class injects NumPy C include directories into the Cython compilation
    after NumPy was installed and is available. This allows us to not depend on
    a preinstalled NumPy but automatically install NumPy with the
    setup_requires arguement.
    """

    def finalize_options(self):
        """Injects Numpy C include directories.

        Overrides default finalize_options method.
        """
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


ext_modules = [Extension("pybm3d.bm3d",
                         sources=["pybm3d/bm3d.pyx",
                                  "bm3d_src/iio.c",
                                  "bm3d_src/bm3d.cpp",
                                  "bm3d_src/lib_transforms.cpp",
                                  "bm3d_src/utilities.cpp", ],
                         language="c++",
                         libraries=["png", "tiff", "jpeg", "fftw3", "fftw3f"])]

setup(
    name='pybm3d',
    version='0.1.0',
    description='Python wrapper around BM3D',
    author='Eric Jonas',
    author_email='jonas@ericjonas.com',
    url='https://github.com/ericmjonas/pybm3d',
    packages=['pybm3d'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    setup_requires=['numpy>=1.13', ],
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27',
        'numpy>=1.13', ],
    tests_require=[
        'pytest',
        'scikit-image', ],
)
