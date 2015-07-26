# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool
cimport numpy as np
import numpy as np

cdef extern from "../bm3d_src/mt19937ar.h":
    double mt_genrand_res53()

cdef extern from "../bm3d_src/bm3d.h":
    int run_bm3d( const float sigma, vector[float] &img_noisy, 
                  vector[float] &img_basic,
                  vector[float] &img_denoised, 
                  const unsigned width, 
                  const unsigned height, 
                  const unsigned chnls, 
                  const bool useSD_h, 
                  const bool useSD_w, 
                  const unsigned tau_2D_hard, 
                  const unsigned tau_2D_wien, 
                  const unsigned color_space)

cdef extern from "../bm3d_src/utilities.h":
    int save_image(char * name, vector[float] & img, 
                   const unsigned width, 
                   const unsigned height, 
                   const unsigned chnls)


def hello():
    return "Hello World"

def random():
    return mt_genrand_res53()

cpdef float[:, :, :] bm3d(float[:, :, :] input_array, 
                          float sigma, 
                          bool useSD_h = True, 
                          bool useSD_w = True, 
                          str tau_2D_hard = "DCT", 
                          str tau_2D_wien = "DCT"
                          ):
    """
    sigma: value of assumed noise of the noisy image;

    input_array : input image, H x W x channum

    useSD_h (resp. useSD_w): if true, use weight based
    on the standard variation of the 3D group for the
    first (resp. second) step, otherwise use the number
    of non-zero coefficients after Hard Thresholding
    (resp. the norm of Wiener coefficients);
 
    tau_2D_hard (resp. tau_2D_wien): 2D transform to apply
    on every 3D group for the first (resp. second) part.
    Allowed values are 'DCT' and 'BIOR';

    # FIXME : add color space support; right now just RGB
    """

    cdef vector[float] input_image
    cdef vector[float] basic_image
    cdef vector[float] output_image
    cdef vector[float] denoised_image

    height = input_array.shape[0]
    width = input_array.shape[1]
    chnls = input_array.shape[2]
    

    # convert the input image
    input_image.resize(input_array.size)
    pos = 0
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            for k in range(input_array.shape[2]):
                input_image[pos] = input_array[i, j, k]
                pos +=1 

                
    if tau_2D_hard == "DCT":
        tau_2D_hard_i = 4
    elif tau_2D_hard == "BIOR" :
        tau_2D_hard_i = 5
    else:
        raise ValueError("Unknown tau_2d_hard, must be DCT or BIOR")

    if tau_2D_wien == "DCT":
        tau_2D_wien_i = 4
    elif tau_2D_wien == "BIOR" :
        tau_2D_wien_i = 5
    else:
        raise ValueError("Unknown tau_2d_wien, must be DCT or BIOR")

    # FIXME someday we'll have color support
    color_space = 0

    ret = run_bm3d(sigma, input_image, basic_image, output_image, 
                   width, height, chnls, 
                   useSD_h, useSD_w, 
                   tau_2D_hard_i, tau_2D_wien_i, 
                   color_space)
    if ret != 0:
        raise Exception("run_bmd3d returned an error, retval=%d" % ret)

    cdef np.ndarray output_array = np.zeros([height, width, chnls], 
                                            dtype = np.float32)


    pos = 0
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            for k in range(input_array.shape[2]):
                output_array[i, j, k] = output_image[pos]
                pos +=1 
    
    return output_array



