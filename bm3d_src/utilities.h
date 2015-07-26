#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include <vector>
#include <fftw3.h>

//! Read image and check number of channels
int load_image(
    char* name
,   std::vector<float> &img
,   unsigned * width
,   unsigned * height
,   unsigned * chnls
);

//! Write image
int save_image(
    char* name
,   std::vector<float> &img
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
);

//! Add noise
void add_noise(
    const std::vector<float> &img
,   std::vector<float> &img_noisy
,   const float sigma
);

//! Check if a number is a power of 2
bool power_of_2(
    const unsigned n
);

//! Add boundaries by symetry
void symetrize(
    const std::vector<float> &img
,   std::vector<float> &img_sym
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
);

//! Subdivide an image into small sub-images
void sub_divide(
    std::vector<float> &img
,   std::vector<std::vector<float> > &sub_img
,   std::vector<unsigned> &w_table
,   std::vector<unsigned> &h_table
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
,   bool divide
);

//! Compute the PSNR and RMSE between img_1 and img_2
int compute_psnr(
    const std::vector<float> &img_1
,   const std::vector<float> &img_2
,   float *psnr
,   float *rmse
);

//! Compute the difference images between img_1 and img_2
int compute_diff(
    const std::vector<float> &img_1
,   const std::vector<float> &img_2
,   std::vector<float> &img_diff
,   const float sigma
);

//! Transform the color space of the image
int color_space_transform(
    std::vector<float> &img
,   const unsigned color_space
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const bool rgb2yuv
);

//! Look for the closest power of 2 number
int closest_power_of_2(
    const unsigned n
);

//! Estimate sigma on each channel according to the choice of the color_space
int estimate_sigma(
    const float sigma
,   std::vector<float> &sigma_table
,   const unsigned chnls
,   const unsigned color_space
);

//! Initialize a set of indices
void ind_initialize(
    std::vector<unsigned> &ind_set
,   const unsigned max_size
,   const unsigned N
,   const unsigned step
);

//! For convenience
unsigned ind_size(
    const unsigned max_size
,   const unsigned N
,   const unsigned step
);

//! Initialize a 2D fftwf_plan with some parameters
void allocate_plan_2d(
    fftwf_plan* plan
,   const unsigned N
,   const fftwf_r2r_kind kind
,   const unsigned nb
);

//! Initialize a 1D fftwf_plan with some parameters
void allocate_plan_1d(
    fftwf_plan* plan
,   const unsigned N
,   const fftwf_r2r_kind kind
,   const unsigned nb
);

//! Tabulated values of log2(2^n)
unsigned ind_log2(
    const unsigned N
);

//! Tabulated values of 2^N
unsigned ind_pow2(
    const unsigned N
);


#endif // UTILITIES_H_INCLUDED
