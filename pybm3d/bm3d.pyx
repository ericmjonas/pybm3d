# setuptools: language = c++
import numpy as np

from libcpp.vector cimport vector
from libcpp cimport bool
cimport numpy as np


__all__ = ['PARAMS', 'bm3d']


PARAMS = {"YUV": 0,
          "YCBCR": 1,
          "OPP": 2,
          "RGB": 3,
          "DCT": 4,
          "BIOR": 5,
          "HADAMARD": 6,
          "NONE": 7}


cdef extern from "../bm3d_src/bm3d.h":
    int run_bm3d(const float sigma, vector[float] &img_noisy,
                 vector[float] &img_basic,
                 vector[float] &img_denoised,
                 const unsigned width,
                 const unsigned height,
                 const unsigned chnls,
                 const bool useSD_h,
                 const bool useSD_w,
                 const unsigned tau_2D_hard,
                 const unsigned tau_2D_wien,
                 const unsigned color_space,
                 const unsigned patch_size,
                 const bool verbose)


cpdef float[:,:,:] run_bm3d_wrap(
    float[:,:,:] input_array,
    float sigma,
    bool useSD_h=True,
    bool useSD_w=True,
    str tau_2D_hard="DCT",
    str tau_2D_wien="DCT",
    str color_space="YUV",
    int patch_size=0,
    bool verbose=False):
    """
    sigma: value of assumed noise of the noisy image
    patch_size: overrides the default patch size selection.
        patch_size=0: use default behavior
        patch_size>0: size to be used

    input_array : input image, H x W x channum

    useSD_h (resp. useSD_w): if true, use weight based
    on the standard variation of the 3D group for the
    first (resp. second) step, otherwise use the number
    of non-zero coefficients after Hard Thresholding
    (resp. the norm of Wiener coefficients);

    tau_2D_hard (resp. tau_2D_wien): 2D transform to apply
    on every 3D group for the first (resp. second) part.
    Allowed values are 'DCT' and 'BIOR';
    """
    tau_2D_hard_i = PARAMS.get(tau_2D_hard)
    if tau_2D_hard_i is None:
        raise ValueError("Unknown tau_2d_hard, must be 'DCT' or 'BIOR'")

    tau_2D_wien_i = PARAMS.get(tau_2D_wien)
    if tau_2D_wien_i is None:
        raise ValueError("Unknown tau_2d_wien, must be 'DCT' or 'BIOR'")

    color_space_i = PARAMS.get(color_space)
    if color_space_i is None:
        raise ValueError("Unknown color_space, must be 'RGB', 'YUV', 'OPP'"
                         " or 'YCBCR'")

    if patch_size < 0:
        raise ValueError("The patch_size parameter must be 0 (default"
                         " behavior) or larger than 0 (size to be used).")

    cdef vector[float] input_image, basic_image, output_image

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

    ret = run_bm3d(sigma, input_image, basic_image, output_image,
                   width, height, chnls,
                   useSD_h, useSD_w,
                   tau_2D_hard_i, tau_2D_wien_i,
                   color_space_i,
                   patch_size,
                   verbose)
    if ret != 0:
        raise Exception("run_bmd3d returned an error, ret_val=%d" % ret)

    cdef np.ndarray output_array = np.zeros([height, width, chnls],
                                            dtype=np.float32)

    pos = 0
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            for k in range(input_array.shape[2]):
                output_array[i, j, k] = output_image[pos]
                pos +=1

    return output_array


def bm3d(input_array, *args, clip=True, **kwargs):
    """Applies BM3D to the given input_array.

    This function calls the Cython wrapped run_bm3d C function. Before inputs
    are preprocessed:
    1. Convert to type Float
    2. If necessary, add third channel dimension
    """
    input_array = np.array(input_array)
    initial_shape, initial_dtype = input_array.shape, input_array.dtype

    input_array = np.atleast_3d(input_array).astype(np.float32)

    out = run_bm3d_wrap(input_array, *args, **kwargs)
    if clip:
        if np.issubdtype(initial_dtype, np.integer):
            dtype_info = np.iinfo(initial_dtype)
        else:
            dtype_info = np.finfo(initial_dtype)
        out = np.clip(out, dtype_info.min, dtype_info.max)
    out = np.array(out, dtype=initial_dtype).reshape(initial_shape)

    return out
