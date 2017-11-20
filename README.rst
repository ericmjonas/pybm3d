PyBM3D
=======

|license| |unix_build| |code_quality|

    | *So you want to denoise some images, or maybe shrink inside a projected gradient algorithm?*

This Python package provides an interface for the BM3D denoising strategy which is based on enhanced sparse representations in the transform-domain. The enhancement of the sparsity is achieved by grouping similar 2D image fragments (e.g. blocks) into 3D data arrays. Visit the offical BM3D `website <http://www.cs.tut.fi/~foi/GCF-BM3D/>`_ for a detailed explanation, benchmark results and other related works.

The core C implementation of BM3D is largely based on the `work <http://www.ipol.im/pub/art/2012/l-bm3d/>`_ of Marc Lebrun.

Installation
____________
PyBM3D is supported for Linux and OSX and Python 2.7 and 3.6. Please follow the installation instructions:

1. Install `FFTW3 <http://www.fftw.org/>`_:

    a. Linux: ``sudo apt-get update && sudo apt-get install libfftw3-dev``
    b. OSX: ``brew update && brew install fftw``

1. (**Optional**, for multithreading) Install compiler with `OpenMP <http://www.openmp.org/>`_ support:

    a. Linux: ``sudo apt-get update && sudo apt-get install gcc``
    b. OSX: ``brew update && brew install gcc``

2. ``pip install pybm3d``

Example
________
+------------------------------------------------------------------------------+
| **Denoising a RGB color image**                                              |
+------------------------------------------------------------------------------+
| .. code:: python                                                             |
|                                                                              |
|  import numpy as np                                                          |
|  from skimage import data, measure                                           |
|                                                                              |
|  import pybm3d                                                               |
|                                                                              |
|                                                                              |
|  noise_std_dev = 40.0                                                        |
|  img = data.astronaut()                                                      |
|  noise = np.random.normal(scale=noise_std_dev,                               |
|                           size=img.shape).astype(img.dtype)                  |
|                                                                              |
|  noisy_img = img + noise                                                     |
|                                                                              |
|  out = pybm3d.bm3d(noisy_img,                                                |
|                    noise_std_dev,                                            |
|                    verbose=True)                                             |
|                                                                              |
|  noise_psnr = measure.compare_psnr(img, noisy_img)                           |
|  out_psnr = measure.compare_psnr(img, out)                                   |
|                                                                              |
|  print("PSNR of noisy image: ", noise_psnr)                                  |
|  print("PSNR of reconstructed image: ", out_psnr)                            |
|                                                                              |
+------------------------------------------------------------------------------+

Testing
________
PyBM3D is tested and linted using `tox <https://tox.readthedocs.io/en/latest/>`_, *pytest* and *flake8*. In order to run all tests for multiple Python versions do the following:

1. Install supported Python versions (2.7 and 3.6) with `pyenv <https://github.com/pyenv/pyenv>`_

3. ``pip install tox``

2. Run tests by executing:

    a. All test environments: ``tox``
    b. Single environment: ``tox -e env`` with an ``env`` specified in ``tox.ini``

License
________
This project is released under the terms of the `GPL3 license <https://opensource.org/licenses/GPL-3.0>`_.


.. |license| image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
    :target: https://www.gnu.org/licenses/gpl-3.0
    :alt: License: GPL v3

.. |unix_build| image:: https://img.shields.io/travis/ericmjonas/pybm3d.svg?branch=master&style=flat&label=unix%20build
    :target: https://travis-ci.org/ericmjonas/pybm3d/
    :alt: Travis-CI Status

.. |code_quality| image:: https://scrutinizer-ci.com/g/ericmjonas/pybm3d/badges/quality-score.png?b=master
    :target: https://scrutinizer-ci.com/g/ericmjonas/pybm3d/?branch=master
    :alt: Scrutinizer Code Quality
