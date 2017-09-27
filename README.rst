# PyBM3D
[![Build Status](https://travis-ci.org/ericmjonas/pybm3d.svg?branch=master)](https://travis-ci.org/ericmjonas/pybm3d)


> So you want to denoise some images, or maybe shrink inside a projected
> gradient algorithm?

This is a python wrapper around Marc Lebrun's implementation of BM3d

http://www.ipol.im/pub/art/2012/l-bm3d/

License is GPL3 per his original source.

The cython wrappings are minimalistic. 

Example
-------
+------------------------------------------------------------------------------+
| **Denoising a 3-channel color image**                                        |
+------------------------------------------------------------------------------+
| .. code:: python                                                             |
|                                                                              |
|  import numpy as np                                                          |
|  import skimage.data                                                         |
|  import scipy.misc                                                           | 
|                                                                              |
|  import pybm3d                                                               |
|                                                                              |
|                                                                              |
|  noise_std_dev = 40                                                          |
|  img = skimage.data.astronaut().astype(np.float32)                           |
|  noise = np.random.normal(scale=noise_std_dev,                               |
|                           size=img.shape).astype(np.float32)                 |
|                                                                              |
|  noisy_img = img + noise                                                     |
|                                                                              |
|  scipy.misc.imsave("noisy_img.png", noisy_img)                               |
|                                                                              |
|  out = pybm3d.bm3d.bm3d(noisy_img, noise_std_dev)                            |
|  scipy.misc.imsave("out.png", out)                                           |
|                                                                              |
+------------------------------------------------------------------------------+

License
-------
This project is released under the terms of the `GPL3 license <https://opensource.org/licenses/GPL-3.0>`_.
