/*
 * Copyright (c) 2011, Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * @file utilities.cpp
 * @brief Utilities functions
 *
 * @author Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 **/

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#include "mt19937ar.h"
#include "io_png.h"
#include "utilities.h"

#define YUV       0
#define YCBCR     1
#define OPP       2
#define RGB       3

 using namespace std;

 /**
  * @brief Load image, check the number of channels
  *
  * @param name : name of the image to read
  * @param img : vector which will contain the image : R, G and B concatenated
  * @param width, height, chnls : size of the image
  *
  * @return EXIT_SUCCESS if the image has been loaded, EXIT_FAILURE otherwise
  **/
int load_image(
    char* name
,   vector<float> &img
,   unsigned * width
,   unsigned * height
,   unsigned * chnls
){
    //! read input image
	cout << endl << "Read input image...";
	size_t h, w, c;
	float *tmp = NULL;
	tmp = read_png_f32(name, &w, &h, &c);
	if (!tmp)
	{
		cout << "error :: " << name << " not found or not a correct png image" << endl;
		return EXIT_FAILURE;
	}
	cout << "done." << endl;

	//! test if image is really a color image and exclude the alpha channel
	if (c > 2)
	{
	    unsigned k = 0;
	    while (k < w * h && tmp[k] == tmp[w * h + k] && tmp[k] == tmp[2 * w * h + k])
            k++;
        c = (k == w * h ? 1 : 3);
	}

	//! Some image informations
	cout << "image size :" << endl;
	cout << " - width          = " << w << endl;
	cout << " - height         = " << h << endl;
	cout << " - nb of channels = " << c << endl;

	//! Initializations
	*width  = w;
	*height = h;
	*chnls  = c;
	img.resize(w * h * c);
	for (unsigned k = 0; k < w * h * c; k++)
        img[k] = tmp[k];

    return EXIT_SUCCESS;
}

/**
 * @brief write image
 *
 * @param name : path+name+extension of the image
 * @param img : vector which contains the image
 * @param width, height, chnls : size of the image
 *
 * @return EXIT_SUCCESS if the image has been saved, EXIT_FAILURE otherwise
 **/
int save_image(
    char* name
,   std::vector<float> &img
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
){
    //! Allocate Memory
    float* tmp = new float[width * height * chnls];

    //! Check for boundary problems
    for (unsigned k = 0; k < width * height * chnls; k++)
        tmp[k] = (img[k] > 255.0f ? 255.0f : (img[k] < 0.0f ? 0.0f : img[k]));

    if (write_png_f32(name, tmp, width, height, chnls) != 0)
    {
        cout << "... failed to save png image " << name << endl;
        return EXIT_FAILURE;
    }

    //! Free Memory
    delete[] tmp;

    return EXIT_SUCCESS;
}


/**
 * @brief add noise to img
 *
 * @param img : original noise-free image
 * @param img_noisy = img + noise
 * @param sigma : standard deviation of the noise
 *
 * @return none.
 **/
void add_noise(
    const vector<float> &img
,   vector<float> &img_noisy
,   const float sigma
){
    const unsigned size = img.size();
    if (img_noisy.size() != size)
        img_noisy.resize(size);

    mt_init_genrand((unsigned long int) time (NULL) + (unsigned long int) getpid());

    for (unsigned k = 0; k < size; k++)
    {
        double a = mt_genrand_res53();
        double b = mt_genrand_res53();
        double z = (double)(sigma) * sqrt(-2.0 * log(a)) * cos(2.0 * M_PI * b);

        img_noisy[k] =  img[k] + (float) z;
    }
}

/**
 * @brief Check if a number is a power of 2
 **/
bool power_of_2(
    const unsigned n
){
    if (n == 0)
        return false;

    if (n == 1)
        return true;

    if (n % 2 == 0)
        return power_of_2(n / 2);
    else
        return false;
}

/**
 * @brief Add boundaries by symetry
 *
 * @param img : image to symetrize
 * @param img_sym : will contain img with symetrized boundaries
 * @param width, height, chnls : size of img
 * @param N : size of the boundary
 *
 * @return none.
 **/
void symetrize(
    const std::vector<float> &img
,   std::vector<float> &img_sym
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
){
    //! Declaration
    const unsigned w = width + 2 * N;
    const unsigned h = height + 2 * N;

    if (img_sym.size() != w * h * chnls)
        img_sym.resize(w * h * chnls);

    for (unsigned c = 0; c < chnls; c++)
    {
        unsigned dc = c * width * height;
        unsigned dc_2 = c * w * h + N * w + N;

        //! Center of the image
        for (unsigned i = 0; i < height; i++)
            for (unsigned j = 0; j < width; j++, dc++)
                img_sym[dc_2 + i * w + j] = img[dc];

        //! Top and bottom
        dc_2 = c * w * h;
        for (unsigned j = 0; j < w; j++, dc_2++)
            for (unsigned i = 0; i < N; i++)
            {
                img_sym[dc_2 + i * w] = img_sym[dc_2 + (2 * N - i - 1) * w];
                img_sym[dc_2 + (h - i - 1) * w] = img_sym[dc_2 + (h - 2 * N + i) * w];
            }

        //! Right and left
        dc_2 = c * w * h;
        for (unsigned i = 0; i < h; i++)
        {
            const unsigned di = dc_2 + i * w;
            for (unsigned j = 0; j < N; j++)
            {
                img_sym[di + j] = img_sym[di + 2 * N - j - 1];
                img_sym[di + w - j - 1] = img_sym[di + w - 2 * N + j];
            }
        }
    }

    return;
}

/**
 * @brief Subdivide an image into small sub-images
 *
 * @param img : image to subdivide
 * @param sub_images: will contain all sub_images
 * @param w_table, h_table : size of sub-images contained in sub_img
 * @param width, height, chnls: size of img
 * @param divide: if true, sub-divides img into sub_img, else rebuild
 *        img from sub_images
 *
 * @return none.
 **/
void sub_divide(
    vector<float> &img
,   vector<vector<float> > &sub_img
,   vector<unsigned> &w_table
,   vector<unsigned> &h_table
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned N
,   bool divide
){
    //! Add by symetry boundaries to the img
    const unsigned h_b = height + 2 * N;
    const unsigned w_b = width  + 2 * N;
    vector<float> img_sym;
    if (divide)
        symetrize(img, img_sym, width, height, chnls, N);

    //! Obtain nb of sub_images in row and column
    unsigned w_small = width;
    unsigned h_small = height;
    unsigned n = sub_img.size();
    unsigned nw = 1;
    unsigned nh = 1;
    while (n > 1)
    {
        if (w_small > h_small)
        {
            w_small = (unsigned) floor((float) w_small * 0.5f);
            nw *= 2;
        }
        else
        {
            h_small = (unsigned) floor((float) h_small * 0.5f);
            nh *=2;
        }
        n /= 2;
    }

    //! As the image may don't have power of 2 dimensions, it may exist a boundary
    const unsigned h_bound = (nh > 1 ? height - (nh - 1) * h_small : h_small);
    const unsigned w_bound = (nw > 1 ? width  - (nw - 1) * w_small : w_small);

    if (divide) //! Subdivides the image in small parts
    {
        for (unsigned i = 0; i < nh; i++)
            for (unsigned j = 0; j < nw; j++)
            {
                const unsigned k = i * nw + j;
                const unsigned h = (i == nh - 1 ? h_bound : h_small) + 2 * N;
                const unsigned w = (j == nw - 1 ? w_bound : w_small) + 2 * N;
                h_table[k] = h;
                w_table [k] = w;
                sub_img[k].resize(w * h * chnls);

                for (unsigned c = 0; c < chnls; c++)
                {
                    unsigned dc_2 = c * w_b * h_b + i * h_small * w_b + j * w_small;
                    for (unsigned p = 0; p < h; p++)
                    {
                        unsigned dq = c * w * h + p * w;
                        for (unsigned q = 0; q < w; q++, dq++)
                            sub_img[k][dq] = img_sym[dc_2 + p * w_b + q];
                    }
                }
            }
    }
    else //! Reconstruction of the image
    {
        for (unsigned i = 0; i < nh; i++)
            for (unsigned j = 0; j < nw; j++)
            {
                const unsigned k = i * nw + j;
                const unsigned h = (i == nh - 1 ? h_bound : h_small) + 2 * N;
                const unsigned w = (j == nw - 1 ? w_bound : w_small) + 2 * N;
                for (unsigned c = 0; c < chnls; c++)
                {
                    unsigned dc = c * w * h + N * w + N;
                    unsigned dc_2 = c * width * height + i * h_small * width + j * w_small;
                    for (unsigned p = 0; p < h - 2 * N; p++)
                    {
                        unsigned dq = dc + p * w;
                        for (unsigned q = 0; q < w - 2 * N; q++, dq++)
                            img[dc_2 + p * width + q] = sub_img[k][dq];
                    }
                }
            }
    }
}

/**
 * @brief Compute PSNR and RMSE between img_1 and img_2
 *
 * @param img_1 : pointer to an allocated array of pixels.
 * @param img_2 : pointer to an allocated array of pixels.
 * @param psnr  : will contain the PSNR
 * @param rmse  : will contain the RMSE
 *
 * @return EXIT_FAILURE if both images haven't the same size.
 **/
int compute_psnr(
    const vector<float> &img_1
,   const vector<float> &img_2
,   float *psnr
,   float *rmse
)
{
    if (img_1.size() != img_2.size())
    {
        cout << "Can't compute PSNR & RMSE: images have different sizes: " << endl;
        cout << "img_1 : " << img_1.size() << endl;
        cout << "img_2 : " << img_2.size() << endl;
        return EXIT_FAILURE;
    }

    float tmp = 0.0f;
    for (unsigned k = 0; k < img_1.size(); k++)
        tmp += (img_1[k] - img_2[k]) * (img_1[k] - img_2[k]);

    (*rmse) = sqrtf(tmp / (float) img_1.size());
    (*psnr) = 20.0f * log10f(255.0f / (*rmse));

    return EXIT_SUCCESS;
}

/**
 * @brief Compute a difference image between img_1 and img_2
 **/
int compute_diff(
    const std::vector<float> &img_1
,   const std::vector<float> &img_2
,   std::vector<float> &img_diff
,   const float sigma
){
    if (img_1.size() != img_2.size())
    {
        cout << "Can't compute difference, img_1 and img_2 don't have the same size" << endl;
        cout << "img_1 : " << img_1.size() << endl;
        cout << "img_2 : " << img_2.size() << endl;
        return EXIT_FAILURE;
    }

    const unsigned size = img_1.size();

    if (img_diff.size() != size)
        img_diff.resize(size);

    const float s = 4.0f * sigma;

    for (unsigned k = 0; k < size; k++)
        {
            float value =  (img_1[k] - img_2[k] + s) * 255.0f / (2.0f * s);
            img_diff[k] = (value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value));
        }

    return EXIT_SUCCESS;
}

/**
 * @brief Transform the color space of the image
 *
 * @param img: image to transform
 * @param color_space: choice between OPP, YUV, YCbCr, RGB
 * @param width, height, chnls: size of img
 * @param rgb2yuv: if true, transform the color space
 *        from RGB to YUV, otherwise do the inverse
 *
 * @return EXIT_FAILURE if color_space has not expected
 *         type, otherwise return EXIT_SUCCESS.
 **/
int color_space_transform(
    vector<float> &img
,   const unsigned color_space
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const bool rgb2yuv
){
    if (chnls == 1 || color_space == RGB)
        return EXIT_SUCCESS;

    //! Declarations
    vector<float> tmp;
    tmp.resize(chnls * width * height);
    const unsigned red   = 0;
    const unsigned green = width * height;
    const unsigned blue  = width * height * 2;

    //! Transformations depending on the mode
    if (color_space == YUV)
    {
        if (rgb2yuv)
        {
        #pragma omp parallel for
            for (unsigned k = 0; k < width * height; k++)
            {
                //! Y
                tmp[k + red  ] =  0.299f   * img[k + red] + 0.587f   * img[k + green] + 0.114f   * img[k + blue];
                //! U
                tmp[k + green] = -0.14713f * img[k + red] - 0.28886f * img[k + green] + 0.436f   * img[k + blue];
                //! V
                tmp[k + blue ] =  0.615f   * img[k + red] - 0.51498f * img[k + green] - 0.10001f * img[k + blue];
            }
        }
        else
        {
        #pragma omp parallel for
            for (unsigned k = 0; k < width * height; k++)
            {
                //! Red   channel
                tmp[k + red  ] = img[k + red] + 1.13983f * img[k + blue];
                //! Green channel
                tmp[k + green] = img[k + red] - 0.39465f * img[k + green] - 0.5806f * img[k + blue];
                //! Blue  channel
                tmp[k + blue ] = img[k + red] + 2.03211f * img[k + green];
            }
        }
    }
    else if (color_space == YCBCR)
    {
        if (rgb2yuv)
        {
        #pragma omp parallel for
            for (unsigned k = 0; k < width * height; k++)
            {
                //! Y
                tmp[k + red  ] =  0.299f * img[k + red] + 0.587f * img[k + green] + 0.114f * img[k + blue];
                //! U
                tmp[k + green] = -0.169f * img[k + red] - 0.331f * img[k + green] + 0.500f * img[k + blue];
                //! V
                tmp[k + blue ] =  0.500f * img[k + red] - 0.419f * img[k + green] - 0.081f * img[k + blue];
            }
        }
        else
        {
        #pragma omp parallel for
            for (unsigned k = 0; k < width * height; k++)
            {
                //! Red   channel
                tmp[k + red  ] = 1.000f * img[k + red] + 0.000f * img[k + green] + 1.402f * img[k + blue];
                //! Green channel
                tmp[k + green] = 1.000f * img[k + red] - 0.344f * img[k + green] - 0.714f * img[k + blue];
                //! Blue  channel
                tmp[k + blue ] = 1.000f * img[k + red] + 1.772f * img[k + green] + 0.000f * img[k + blue];
            }
        }
    }
    else if (color_space == OPP)
    {
        if (rgb2yuv)
        {
        #pragma omp parallel for
            for (unsigned k = 0; k < width * height; k++)
            {
                //! Y
                tmp[k + red  ] =  0.333f * img[k + red] + 0.333f * img[k + green] + 0.333f * img[k + blue];
                //! U
                tmp[k + green] =  0.500f * img[k + red] + 0.000f * img[k + green] - 0.500f * img[k + blue];
                //! V
                tmp[k + blue ] =  0.250f * img[k + red] - 0.500f * img[k + green] + 0.250f * img[k + blue];
            }
        }
        else
        {
        #pragma omp parallel for
            for (unsigned k = 0; k < width * height; k++)
            {
                //! Red   channel
                tmp[k + red  ] = 1.0f * img[k + red] + 1.0f * img[k + green] + 0.666f * img[k + blue];
                //! Green cha
                tmp[k + green] = 1.0f * img[k + red] + 0.0f * img[k + green] - 1.333f * img[k + blue];
                //! Blue  cha
                tmp[k + blue ] = 1.0f * img[k + red] - 1.0f * img[k + green] + 0.666f * img[k + blue];
            }
        }
    }
    else
    {
        cout << "Wrong type of transform. Must be OPP, YUV, or YCbCr!!" << endl;
        return EXIT_FAILURE;
    }

    #pragma omp parallel for
        for (unsigned k = 0; k < width * height * chnls; k++)
            img[k] = tmp[k];

    return EXIT_SUCCESS;
}

/**
 * @brief Look for the closest power of 2 number
 *
 * @param n: number
 *
 * @return the closest power of 2 lower or equal to n
 **/
int closest_power_of_2(
    const unsigned n
){
    unsigned r = 1;
    while (r * 2 <= n)
        r *= 2;

    return r;
}

/**
 * @brief Estimate sigma on each channel according to
 *        the choice of the color_space.
 *
 * @param sigma: estimated standard deviation of the noise;
 * @param sigma_Y : noise on the first channel;
 * @param sigma_U : (if chnls > 1) noise on the second channel;
 * @param sigma_V : (if chnls > 1) noise on the third channel;
 * @param chnls : number of channels of the image;
 * @param color_space : choice between OPP, YUV, YCbCr. If not
 *        then we assume that we're still in RGB space.
 *
 * @return EXIT_FAILURE if color_space has not expected
 *         type, otherwise return EXIT_SUCCESS.
 **/
int estimate_sigma(
    const float sigma
,   std::vector<float> &sigma_table
,   const unsigned chnls
,   const unsigned color_space
){
    if (chnls == 1)
        sigma_table[0] = sigma;
    else
    {
        if (color_space == YUV)
        {
            //! Y
            sigma_table[0] = sqrtf(0.299f * 0.299f + 0.587f * 0.587f + 0.114f * 0.114f) * sigma;
            //! U
            sigma_table[1] = sqrtf(0.14713f * 0.14713f + 0.28886f * 0.28886f + 0.436f * 0.436f) * sigma;
            //! V
            sigma_table[2] = sqrtf(0.615f * 0.615f + 0.51498f * 0.51498f + 0.10001f * 0.10001f) * sigma;
        }
        else if (color_space == YCBCR)
        {
            //! Y
            sigma_table[0] = sqrtf(0.299f * 0.299f + 0.587f * 0.587f + 0.114f * 0.114f) * sigma;
            //! U
            sigma_table[1] = sqrtf(0.169f * 0.169f + 0.331f * 0.331f + 0.500f * 0.500f) * sigma;
            //! V
            sigma_table[2] = sqrtf(0.500f * 0.500f + 0.419f * 0.419f + 0.081f * 0.081f) * sigma;
        }
        else if (color_space == OPP)
        {
            //! Y
            sigma_table[0] = sqrtf(0.333f * 0.333f + 0.333f * 0.333f + 0.333f * 0.333f) * sigma;
            //! U
            sigma_table[1] = sqrtf(0.5f * 0.5f + 0.0f * 0.0f + 0.5f * 0.5f) * sigma;
            //! V
            sigma_table[2] = sqrtf(0.25f * 0.25f + 0.5f * 0.5f + 0.25f * 0.25f) * sigma;
        }
        else if (color_space == RGB)
        {
            //! Y
            sigma_table[0] = sigma;
            //! U
            sigma_table[1] = sigma;
            //! V
            sigma_table[2] = sigma;
        }
        else
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/**
 * @brief Initialize a set of indices.
 *
 * @param ind_set: will contain the set of indices;
 * @param max_size: indices can't go over this size;
 * @param N : boundary;
 * @param step: step between two indices.
 *
 * @return none.
 **/
void ind_initialize(
    vector<unsigned> &ind_set
,   const unsigned max_size
,   const unsigned N
,   const unsigned step
){
    ind_set.clear();
    unsigned ind = N;
    while (ind < max_size - N)
    {
        ind_set.push_back(ind);
        ind += step;
    }
    if (ind_set.back() < max_size - N - 1)
        ind_set.push_back(max_size - N - 1);
}

/**
 * @brief For convenience. Estimate the size of the ind_set vector built
 *        with the function ind_initialize().
 *
 * @return size of ind_set vector built in ind_initialize().
 **/
unsigned ind_size(
    const unsigned max_size
,   const unsigned N
,   const unsigned step
){
    unsigned ind = N;
    unsigned k = 0;
    while (ind < max_size - N)
    {
        k++;
        ind += step;
    }
    if (ind - step < max_size - N - 1)
        k++;

    return k;
}

/**
 * @brief Initialize a 2D fftwf_plan with some parameters
 *
 * @param plan: fftwf_plan to allocate;
 * @param N: size of the patch to apply the 2D transform;
 * @param kind: forward or backward;
 * @param nb: number of 2D transform which will be processed.
 *
 * @return none.
 **/
void allocate_plan_2d(
    fftwf_plan* plan
,   const unsigned N
,   const fftwf_r2r_kind kind
,   const unsigned nb
){
    int            nb_table[2]   = {N, N};
    int            nembed[2]     = {N, N};
    fftwf_r2r_kind kind_table[2] = {kind, kind};

    float* vec = (float*) fftwf_malloc(N * N * nb * sizeof(float));
    (*plan) = fftwf_plan_many_r2r(2, nb_table, nb, vec, nembed, 1, N * N, vec,
                                  nembed, 1, N * N, kind_table, FFTW_ESTIMATE);

    fftwf_free(vec);
}

/**
 * @brief Initialize a 1D fftwf_plan with some parameters
 *
 * @param plan: fftwf_plan to allocate;
 * @param N: size of the vector to apply the 1D transform;
 * @param kind: forward or backward;
 * @param nb: number of 1D transform which will be processed.
 *
 * @return none.
 **/
void allocate_plan_1d(
    fftwf_plan* plan
,   const unsigned N
,   const fftwf_r2r_kind kind
,   const unsigned nb
){
    int nb_table[1] = {N};
    int nembed[1]   = {N * nb};
    fftwf_r2r_kind kind_table[1] = {kind};

    float* vec = (float*) fftwf_malloc(N * nb * sizeof(float));
    (*plan) = fftwf_plan_many_r2r(1, nb_table, nb, vec, nembed, 1, N, vec,
                                  nembed, 1, N, kind_table, FFTW_ESTIMATE);
    fftwf_free(vec);
}

/**
 * @brief tabulated values of log2(N), where N = 2 ^ n.
 *
 * @param N : must be a power of 2 smaller than 64
 *
 * @return n = log2(N)
 **/
unsigned ind_log2(
    const unsigned N
){
    return (N == 1  ? 0 :
           (N == 2  ? 1 :
           (N == 4  ? 2 :
           (N == 8  ? 3 :
           (N == 16 ? 4 :
           (N == 32 ? 5 : 6) ) ) ) ) );
}

/**
 * @brief tabulated values of log2(N), where N = 2 ^ n.
 *
 * @param N : must be a power of 2 smaller than 64
 *
 * @return n = 2 ^ N
 **/
unsigned ind_pow2(
    const unsigned N
){
    return (N == 0 ? 1  :
           (N == 1 ? 2  :
           (N == 2 ? 4  :
           (N == 3 ? 8  :
           (N == 4 ? 16 :
           (N == 5 ? 32 : 64) ) ) ) ) );
}
