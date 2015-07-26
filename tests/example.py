from nose.tools import * 
import numpy as np
import skimage.data
import scipy.misc

import sys


import pybm3d



x  = skimage.data.camera().astype(np.float32)
x.shape = x.shape[0], x.shape[1], 1

noise = np.random.normal(0, 40, x.shape).astype(np.float32)
x = x + noise

scipy.misc.imsave("foo.png", x[:, :, 0] ) # / np.max(x))
z = pybm3d.bm3d.bm3d(x, 40.0) 

scipy.misc.imsave("out.png", z[:, :, 0] ) # / np.max(x))
