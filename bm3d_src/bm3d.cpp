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
 * @file bm3d.cpp
 * @brief BM3D denoising functions
 *
 * @author Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 **/

#include <iostream>
#include <algorithm>
#include <math.h>

#include "bm3d.h"
#include "utilities.h"
#include "lib_transforms.h"

#define SQRT2     1.414213562373095
#define SQRT2_INV 0.7071067811865475
#define YUV       0
#define YCBCR     1
#define OPP       2
#define RGB       3
#define DCT       4
#define BIOR      5
#define HADAMARD  6

#ifdef _OPENMP
    #include <omp.h>
#endif

 using namespace std;

 bool ComparaisonFirst(pair<float,unsigned> pair1, pair<float,unsigned> pair2)
{
	return pair1.first < pair2.first;
}

/** ----------------- **/
/** - Main function - **/
/** ----------------- **/
/**
 * @brief run BM3D process. Depending on if OpenMP is used or not,
 *        and on the number of available threads, it divides the noisy
 *        image in sub_images, to process them in parallel.
 *
 * @param sigma: value of assumed noise of the noisy image;
 * @param img_noisy: noisy image;
 * @param img_basic: will be the basic estimation after the 1st step
 * @param img_denoised: will be the denoised final image;
 * @param width, height, chnls: size of the image;
 * @param useSD_h (resp. useSD_w): if true, use weight based
 *        on the standard variation of the 3D group for the
 *        first (resp. second) step, otherwise use the number
 *        of non-zero coefficients after Hard Thresholding
 *        (resp. the norm of Wiener coefficients);
 * @param tau_2D_hard (resp. tau_2D_wien): 2D transform to apply
 *        on every 3D group for the first (resp. second) part.
 *        Allowed values are DCT and BIOR;
 * @param color_space: Transformation from RGB to YUV. Allowed
 *        values are RGB (do nothing), YUV, YCBCR and OPP.
 *
 * @return EXIT_FAILURE if color_space has not expected
 *         type, otherwise return EXIT_SUCCESS.
 **/
int run_bm3d(
    const float sigma
,   vector<float> &img_noisy
,   vector<float> &img_basic
,   vector<float> &img_denoised
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const bool useSD_h
,   const bool useSD_w
,   const unsigned tau_2D_hard
,   const unsigned tau_2D_wien
,   const unsigned color_space
){
    //! Parameters
    const unsigned nHard = 16; //! Half size of the search window
    const unsigned nWien = 16; //! Half size of the search window
    const unsigned kHard = (tau_2D_hard == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_hard == BIOR
    const unsigned kWien = (tau_2D_wien == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_wien == BIOR
    const unsigned NHard = 16; //! Must be a power of 2
    const unsigned NWien = 32; //! Must be a power of 2
    const unsigned pHard = 3;
    const unsigned pWien = 3;

    //! Check memory allocation
    if (img_basic.size() != img_noisy.size())
        img_basic.resize(img_noisy.size());
    if (img_denoised.size() != img_noisy.size())
        img_denoised.resize(img_noisy.size());

    //! Transformation to YUV color space
    if (color_space_transform(img_noisy, color_space, width, height, chnls, true)
        != EXIT_SUCCESS) return EXIT_FAILURE;

    //! Check if OpenMP is used or if number of cores of the computer is > 1
    unsigned nb_threads = 1;

#ifdef _OPENMP
    cout << "Open MP used" << endl;
    nb_threads = omp_get_num_procs();

    //! In case where the number of processors isn't a power of 2
    if (!power_of_2(nb_threads))
        nb_threads = closest_power_of_2(nb_threads);
#endif

    cout << endl << "Number of threads which will be used: " << nb_threads;
#ifdef _OPENMP
    cout << " (real available cores: " << omp_get_num_procs() << ")" << endl;
#endif

    //! Allocate plan for FFTW library
    fftwf_plan plan_2d_for_1[nb_threads];
    fftwf_plan plan_2d_for_2[nb_threads];
    fftwf_plan plan_2d_inv[nb_threads];

    //! In the simple case
    if (nb_threads == 1)
    {
        //! Add boundaries and symetrize them
        const unsigned h_b = height + 2 * nHard;
        const unsigned w_b = width  + 2 * nHard;
        vector<float> img_sym_noisy, img_sym_basic, img_sym_denoised;
        symetrize(img_noisy, img_sym_noisy, width, height, chnls, nHard);

        //! Allocating Plan for FFTW process
        if (tau_2D_hard == DCT)
        {
            const unsigned nb_cols = ind_size(w_b - kHard + 1, nHard, pHard);
            allocate_plan_2d(&plan_2d_for_1[0], kHard, FFTW_REDFT10,
                                                        w_b * (2 * nHard + 1) * chnls);
            allocate_plan_2d(&plan_2d_for_2[0], kHard, FFTW_REDFT10,
                                                        w_b * pHard * chnls);
            allocate_plan_2d(&plan_2d_inv  [0], kHard, FFTW_REDFT01,
                                                        NHard * nb_cols * chnls);
        }

        //! Denoising, 1st Step
        cout << "step 1...";
        bm3d_1st_step(sigma, img_sym_noisy, img_sym_basic, w_b, h_b, chnls, nHard,
                      kHard, NHard, pHard, useSD_h, color_space, tau_2D_hard,
                      &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
        cout << "done." << endl;

        //! To avoid boundaries problem
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc_b = c * w_b * h_b + nHard * w_b + nHard;
            unsigned dc = c * width * height;
            for (unsigned i = 0; i < height; i++)
                for (unsigned j = 0; j < width; j++, dc++)
                    img_basic[dc] = img_sym_basic[dc_b + i * w_b + j];
        }
        symetrize(img_basic, img_sym_basic, width, height, chnls, nHard);

        //! Allocating Plan for FFTW process
        if (tau_2D_wien == DCT)
        {
            const unsigned nb_cols = ind_size(w_b - kWien + 1, nWien, pWien);
            allocate_plan_2d(&plan_2d_for_1[0], kWien, FFTW_REDFT10,
                                                        w_b * (2 * nWien + 1) * chnls);
            allocate_plan_2d(&plan_2d_for_2[0], kWien, FFTW_REDFT10,
                                                        w_b * pWien * chnls);
            allocate_plan_2d(&plan_2d_inv  [0], kWien, FFTW_REDFT01,
                                                        NWien * nb_cols * chnls);
        }

        //! Denoising, 2nd Step
        cout << "step 2...";
        bm3d_2nd_step(sigma, img_sym_noisy, img_sym_basic, img_sym_denoised,
                w_b, h_b, chnls, nWien, kWien, NWien, pWien, useSD_w, color_space,
                tau_2D_wien, &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
        cout << "done." << endl;

        //! Obtention of img_denoised
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc_b = c * w_b * h_b + nWien * w_b + nWien;
            unsigned dc = c * width * height;
            for (unsigned i = 0; i < height; i++)
                for (unsigned j = 0; j < width; j++, dc++)
                    img_denoised[dc] = img_sym_denoised[dc_b + i * w_b + j];
        }
    }
    //! If more than 1 threads are used
    else
    {
        //! Cut the image in nb_threads parts
        vector<vector<float> > sub_noisy(nb_threads);
        vector<vector<float> > sub_basic(nb_threads);
        vector<vector<float> > sub_denoised(nb_threads);
        vector<unsigned> h_table(nb_threads);
        vector<unsigned> w_table(nb_threads);
        sub_divide(img_noisy, sub_noisy, w_table, h_table, width, height, chnls,
                                                                        2 * nWien, true);

        //! Allocating Plan for FFTW process
        if (tau_2D_hard == DCT)
            for (unsigned n = 0; n < nb_threads; n++)
            {
                const unsigned nb_cols = ind_size(w_table[n] - kHard + 1, nHard, pHard);
                allocate_plan_2d(&plan_2d_for_1[n], kHard, FFTW_REDFT10,
                                                w_table[n] * (2 * nHard + 1) * chnls);
                allocate_plan_2d(&plan_2d_for_2[n], kHard, FFTW_REDFT10,
                                                w_table[n] * pHard * chnls);
                allocate_plan_2d(&plan_2d_inv  [n], kHard, FFTW_REDFT01,
                                                NHard * nb_cols * chnls);
            }

        //! denoising : 1st Step
        cout << "step 1...";
        #pragma omp parallel shared(sub_noisy, sub_basic, w_table, h_table, \
                                    plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
        {
            #pragma omp for schedule(dynamic) nowait
            for (unsigned n = 0; n < nb_threads; n++)
            {
                bm3d_1st_step(sigma, sub_noisy[n], sub_basic[n], w_table[n],
                              h_table[n], chnls, nHard, kHard, NHard, pHard, useSD_h,
                              color_space, tau_2D_hard, &plan_2d_for_1[n],
                              &plan_2d_for_2[n], &plan_2d_inv[n]);
            }
        }
        cout << "done." << endl;

        sub_divide(img_basic, sub_basic, w_table, h_table,
                                                width, height, chnls, 2 * nHard, false);

        sub_divide(img_basic, sub_basic, w_table, h_table, width, height, chnls,
                                                                        2 * nHard, true);

        //! Allocating Plan for FFTW process
        if (tau_2D_wien == DCT)
            for (unsigned n = 0; n < nb_threads; n++)
            {
                const unsigned nb_cols = ind_size(w_table[n] - kWien + 1, nWien, pWien);
                allocate_plan_2d(&plan_2d_for_1[n], kWien, FFTW_REDFT10,
                                                w_table[n] * (2 * nWien + 1) * chnls);
                allocate_plan_2d(&plan_2d_for_2[n], kWien, FFTW_REDFT10,
                                                w_table[n] * pWien * chnls);
                allocate_plan_2d(&plan_2d_inv  [n], kWien, FFTW_REDFT01,
                                                NWien * nb_cols * chnls);
            }

        //! Denoising: 2nd Step
        cout << "step 2...";
        #pragma omp parallel shared(sub_noisy, sub_basic, sub_denoised,  w_table, \
                                    h_table, plan_2d_for_1, plan_2d_for_2,  \
                                    plan_2d_inv)
        {
            #pragma omp for schedule(dynamic) nowait
            for (unsigned n = 0; n < nb_threads; n++)
            {
                bm3d_2nd_step(sigma, sub_noisy[n], sub_basic[n], sub_denoised[n],
                              w_table[n], h_table[n], chnls, nWien, kWien, NWien, pWien,
                              useSD_w, color_space, tau_2D_wien, &plan_2d_for_1[n],
                              &plan_2d_for_2[n], &plan_2d_inv[n]);
            }
        }
        cout << "done." << endl;

        //! Reconstruction of the image
        sub_divide(img_denoised, sub_denoised, w_table, h_table,
                                                width, height, chnls, 2 * nWien, false);
    }

    //! Inverse color space transform to RGB
    if (color_space_transform(img_denoised, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_noisy, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;
    if (color_space_transform(img_basic, color_space, width, height, chnls, false)
        != EXIT_SUCCESS) return EXIT_FAILURE;

    //! Free Memory
    if (tau_2D_hard == DCT || tau_2D_wien == DCT)
        for (unsigned n = 0; n < nb_threads; n++)
        {
            fftwf_destroy_plan(plan_2d_for_1[n]);
            fftwf_destroy_plan(plan_2d_for_2[n]);
            fftwf_destroy_plan(plan_2d_inv[n]);
        }
    fftwf_cleanup();

    return EXIT_SUCCESS;
}

/**
 * @brief Run the basic process of BM3D (1st step). The result
 *        is contained in img_basic. The image has boundary, which
 *        are here only for block-matching and doesn't need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to denoise;
 * @param img_noisy: noisy image;
 * @param img_basic: will contain the denoised image after the 1st step;
 * @param width, height, chnls : size of img_noisy;
 * @param nHard: size of the boundary around img_noisy;
 * @param useSD: if true, use weight based on the standard variation
 *        of the 3D group for the first step, otherwise use the number
 *        of non-zero coefficients after Hard-thresholding;
 * @param tau_2D: DCT or BIOR;
 * @param plan_2d_for_1, plan_2d_for_2, plan_2d_inv : for convenience. Used
 *        by fftw.
 *
 * @return none.
 **/
void bm3d_1st_step(
    const float sigma
,   vector<float> const& img_noisy
,   vector<float> &img_basic
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned nHard
,   const unsigned kHard
,   const unsigned NHard
,   const unsigned pHard
,   const bool     useSD
,   const unsigned color_space
,   const unsigned tau_2D
,   fftwf_plan *  plan_2d_for_1
,   fftwf_plan *  plan_2d_for_2
,   fftwf_plan *  plan_2d_inv
){
    //! Estimatation of sigma on each channel
    vector<float> sigma_table(chnls);
    if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
        return;

    //! Parameters initialization
    const float    lambdaHard3D = 2.7f;            //! Threshold for Hard Thresholding
    const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma_table[0] < 35.0f ? 2500 : 5000); //! threshold used to determinate similarity between patches

    //! Initialization for convenience
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHard + 1, nHard, pHard);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHard + 1, nHard, pHard);
    const unsigned kHard_2 = kHard * kHard;
    vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<float> hadamard_tmp(NHard);

    //! Check allocation memory
    if (img_basic.size() != img_noisy.size())
        img_basic.resize(img_noisy.size());

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kHard_2);
    vector<float> coef_norm(kHard_2);
    vector<float> coef_norm_inv(kHard_2);
    preProcess(kaiser_window, coef_norm, coef_norm_inv, kHard);

    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! For aggregation part
    vector<float> denominator(width * height * chnls, 0.0f);
    vector<float> numerator  (width * height * chnls, 0.0f);

    //! Precompute Bloc-Matching
    vector<vector<unsigned> > patch_table;
    precompute_BM(patch_table, img_noisy, width, height, kHard, NHard, nHard, pHard, tauMatch);

    //! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
    vector<float> table_2D((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);

    //! Loop on i_r
    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        const unsigned i_r = row_ind[ind_i];

        //! Update of table_2D
        if (tau_2D == DCT)
            dct_2d_process(table_2D, img_noisy, plan_2d_for_1, plan_2d_for_2, nHard,
                           width, height, chnls, kHard, i_r, pHard, coef_norm,
                           row_ind[0], row_ind.back());
        else if (tau_2D == BIOR)
            bior_2d_process(table_2D, img_noisy, nHard, width, height, chnls,
                        kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);

        wx_r_table.clear();
        group_3D_table.clear();

        //! Loop on j_r
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;

            //! Number of similar patches
            const unsigned nSx_r = patch_table[k_r].size();

            //! Build of the 3D group
            vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned ind = patch_table[k_r][n] + (nHard - i_r) * width;
                    for (unsigned k = 0; k < kHard_2; k++)
                        group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                            table_2D[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                }

            //! HT filtering of the 3D group
            vector<float> weight_table(chnls);
            ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                                                    lambdaHard3D, weight_table, !useSD);

            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                    for (unsigned k = 0; k < kHard_2; k++)
                        group_3D_table.push_back(group_3D[n + k * nSx_r +
                                                                    c * kHard_2 * nSx_r]);

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);

        } //! End of loop on j_r

        //!  Apply 2D inverse transform
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * column_ind.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);

        //! Registration of the weighted estimation
        unsigned dec = 0;
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            const unsigned j_r   = column_ind[ind_j];
            const unsigned k_r   = i_r * width + j_r;
            const unsigned nSx_r = patch_table[k_r].size();
            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned k = patch_table[k_r][n] + c * width * height;
                    for (unsigned p = 0; p < kHard; p++)
                        for (unsigned q = 0; q < kHard; q++)
                        {
                            const unsigned ind = k + p * width + q;
                            numerator[ind] += kaiser_window[p * kHard + q]
                                            * wx_r_table[c + ind_j * chnls]
                                            * group_3D_table[p * kHard + q + n * kHard_2
                                                  + c * kHard_2 * nSx_r + dec];
                            denominator[ind] += kaiser_window[p * kHard + q]
                                            * wx_r_table[c + ind_j * chnls];
                        }
                }
            }
            dec += nSx_r * chnls * kHard_2;
        }

    } //! End of loop on i_r

    //! Final reconstruction
    for (unsigned k = 0; k < width * height * chnls; k++)
        img_basic[k] = numerator[k] / denominator[k];
}

/**
 * @brief Run the final process of BM3D (2nd step). The result
 *        is contained in img_denoised. The image has boundary, which
 *        are here only for block-matching and doesn't need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to denoise;
 * @param img_noisy: noisy image;
 * @param img_basic: contains the denoised image after the 1st step;
 * @param img_denoised: will contain the final estimate of the denoised
 *        image after the second step;
 * @param width, height, chnls : size of img_noisy;
 * @param nWien: size of the boundary around img_noisy;
 * @param useSD: if true, use weight based on the standard variation
 *        of the 3D group for the second step, otherwise use the norm
 *        of Wiener coefficients of the 3D group;
 * @param tau_2D: DCT or BIOR.
 *
 * @return none.
 **/
void bm3d_2nd_step(
    const float sigma
,   vector<float> const& img_noisy
,   vector<float> const& img_basic
,   vector<float> &img_denoised
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned nWien
,   const unsigned kWien
,   const unsigned NWien
,   const unsigned pWien
,   const bool     useSD
,   const unsigned color_space
,   const unsigned tau_2D
,   fftwf_plan *  plan_2d_for_1
,   fftwf_plan *  plan_2d_for_2
,   fftwf_plan *  plan_2d_inv
){
    //! Estimatation of sigma on each channel
    vector<float> sigma_table(chnls);
    if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
        return;

    //! Parameters initialization
    const float tauMatch = (sigma_table[0] < 35.0f ? 400 : 3500); //! threshold used to determinate similarity between patches

    //! Initialization for convenience
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kWien + 1, nWien, pWien);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kWien + 1, nWien, pWien);
    const unsigned kWien_2 = kWien * kWien;
    vector<float> group_3D_table(chnls * kWien_2 * NWien * column_ind.size());
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<float> tmp(NWien);

    //! Check allocation memory
    if (img_denoised.size() != img_noisy.size())
        img_denoised.resize(img_noisy.size());

    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    vector<float> kaiser_window(kWien_2);
    vector<float> coef_norm(kWien_2);
    vector<float> coef_norm_inv(kWien_2);
    preProcess(kaiser_window, coef_norm, coef_norm_inv, kWien);

    //! For aggregation part
    vector<float> denominator(width * height * chnls, 0.0f);
    vector<float> numerator  (width * height * chnls, 0.0f);

    //! Precompute Bloc-Matching
    vector<vector<unsigned> > patch_table;
    precompute_BM(patch_table, img_basic, width, height, kWien, NWien, nWien, pWien, tauMatch);

    //! Preprocessing of Bior table
    vector<float> lpd, hpd, lpr, hpr;
    bior15_coef(lpd, hpd, lpr, hpr);

    //! DCT_table_2D[p * N + q + (i * width + j) * kWien_2 + c * (2 * ns + 1) * width * kWien_2]
    vector<float> table_2D_img((2 * nWien + 1) * width * chnls * kWien_2, 0.0f);
    vector<float> table_2D_est((2 * nWien + 1) * width * chnls * kWien_2, 0.0f);

    //! Loop on i_r
    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        const unsigned i_r = row_ind[ind_i];

        //! Update of DCT_table_2D
        if (tau_2D == DCT)
        {
            dct_2d_process(table_2D_img, img_noisy, plan_2d_for_1, plan_2d_for_2,
                           nWien, width, height, chnls, kWien, i_r, pWien, coef_norm,
                           row_ind[0], row_ind.back());
            dct_2d_process(table_2D_est, img_basic, plan_2d_for_1, plan_2d_for_2,
                           nWien, width, height, chnls, kWien, i_r, pWien, coef_norm,
                           row_ind[0], row_ind.back());
        }
        else if (tau_2D == BIOR)
        {
            bior_2d_process(table_2D_img, img_noisy, nWien, width, height,
                chnls, kWien, i_r, pWien, row_ind[0], row_ind.back(), lpd, hpd);
            bior_2d_process(table_2D_est, img_basic, nWien, width, height,
                chnls, kWien, i_r, pWien, row_ind[0], row_ind.back(), lpd, hpd);
        }

        wx_r_table.clear();
        group_3D_table.clear();

        //! Loop on j_r
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //! Initialization
            const unsigned j_r = column_ind[ind_j];
            const unsigned k_r = i_r * width + j_r;

            //! Number of similar patches
            const unsigned nSx_r = patch_table[k_r].size();

            //! Build of the 3D group
            vector<float> group_3D_est(chnls * nSx_r * kWien_2, 0.0f);
            vector<float> group_3D_img(chnls * nSx_r * kWien_2, 0.0f);
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned ind = patch_table[k_r][n] + (nWien - i_r) * width;
                    for (unsigned k = 0; k < kWien_2; k++)
                    {
                        group_3D_est[n + k * nSx_r + c * kWien_2 * nSx_r] =
                            table_2D_est[k + ind * kWien_2 + c * kWien_2 * (2 * nWien + 1) * width];
                        group_3D_img[n + k * nSx_r + c * kWien_2 * nSx_r] =
                            table_2D_img[k + ind * kWien_2 + c * kWien_2 * (2 * nWien + 1) * width];
                    }
                }

            //! Wiener filtering of the 3D group
            vector<float> weight_table(chnls);
            wiener_filtering_hadamard(group_3D_img, group_3D_est, tmp, nSx_r, kWien,
                                            chnls, sigma_table, weight_table, !useSD);

            //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D_est, nSx_r, kWien, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                    for (unsigned k = 0; k < kWien_2; k++)
                        group_3D_table.push_back(group_3D_est[n + k * nSx_r + c * kWien_2 * nSx_r]);

            //! Save weighting
            for (unsigned c = 0; c < chnls; c++)
                wx_r_table.push_back(weight_table[c]);

        } //! End of loop on j_r

        //!  Apply 2D dct inverse
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kWien, NWien * chnls * column_ind.size(),
                           coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kWien, lpr, hpr);

        //! Registration of the weighted estimation
        unsigned dec = 0;
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            const unsigned j_r   = column_ind[ind_j];
            const unsigned k_r   = i_r * width + j_r;
            const unsigned nSx_r = patch_table[k_r].size();
            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned k = patch_table[k_r][n] + c * width * height;
                    for (unsigned p = 0; p < kWien; p++)
                        for (unsigned q = 0; q < kWien; q++)
                        {
                            const unsigned ind = k + p * width + q;
                            numerator[ind] += kaiser_window[p * kWien + q]
                                            * wx_r_table[c + ind_j * chnls]
                                            * group_3D_table[p * kWien + q + n * kWien_2
                                                  + c * kWien_2 * nSx_r + dec];
                            denominator[ind] += kaiser_window[p * kWien + q]
                                            * wx_r_table[c + ind_j * chnls];
                        }
                }
            }
            dec += nSx_r * chnls * kWien_2;
        }

    } //! End of loop on i_r

    //! Final reconstruction
    for (unsigned k = 0; k < width * height * chnls; k++)
        img_denoised[k] = numerator[k] / denominator[k];
}

/**
 * @brief Precompute a 2D DCT transform on all patches contained in
 *        a part of the image.
 *
 * @param DCT_table_2D : will contain the 2d DCT transform for all
 *        chosen patches;
 * @param img : image on which the 2d DCT will be processed;
 * @param plan_1, plan_2 : for convenience. Used by fftw;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW);
 * @param i_r: current index of the reference patches;
 * @param step: space in pixels between two references patches;
 * @param coef_norm : normalization coefficients of the 2D DCT;
 * @param i_min (resp. i_max) : minimum (resp. maximum) value
 *        for i_r. In this case the whole 2d transform is applied
 *        on every patches. Otherwise the precomputed 2d DCT is re-used
 *        without processing it.
 **/
void dct_2d_process(
    vector<float> &DCT_table_2D
,   vector<float> const& img
,   fftwf_plan * plan_1
,   fftwf_plan * plan_2
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned step
,   vector<float> const& coef_norm
,   const unsigned i_min
,   const unsigned i_max
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned size = chnls * kHW_2 * width * (2 * nHW + 1);

    //! If i_r == ns, then we have to process all DCT
    if (i_r == i_min || i_r == i_max)
    {
        //! Allocating Memory
        float* vec = (float*) fftwf_malloc(size * sizeof(float));
        float* dct = (float*) fftwf_malloc(size * sizeof(float));

        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc = c * width * height;
            const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
            for (unsigned i = 0; i < 2 * nHW + 1; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned p = 0; p < kHW; p++)
                        for (unsigned q = 0; q < kHW; q++)
                            vec[p * kHW + q + dc_p + (i * width + j) * kHW_2] =
                                img[dc + (i_r + i - nHW + p) * width + j + q];
        }

        //! Process of all DCTs
        fftwf_execute_r2r(*plan_1, vec, dct);
        fftwf_free(vec);

        //! Getting the result
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc   = c * kHW_2 * width * (2 * nHW + 1);
            const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
            for (unsigned i = 0; i < 2 * nHW + 1; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned k = 0; k < kHW_2; k++)
                        DCT_table_2D[dc + (i * width + j) * kHW_2 + k] =
                            dct[dc_p + (i * width + j) * kHW_2 + k] * coef_norm[k];
        }
        fftwf_free(dct);
    }
    else
    {
        const unsigned ds = step * width * kHW_2;

        //! Re-use of DCT already processed
        for (unsigned c = 0; c < chnls; c++)
        {
            unsigned dc = c * width * (2 * nHW + 1) * kHW_2;
            for (unsigned i = 0; i < 2 * nHW + 1 - step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned k = 0; k < kHW_2; k++)
                        DCT_table_2D[k + (i * width + j) * kHW_2 + dc] =
                            DCT_table_2D[k + (i * width + j) * kHW_2 + dc + ds];
        }

        //! Compute the new DCT
        float* vec = (float*) fftwf_malloc(chnls * kHW_2 * step * width * sizeof(float));
        float* dct = (float*) fftwf_malloc(chnls * kHW_2 * step * width * sizeof(float));

        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc   = c * width * height;
            const unsigned dc_p = c * kHW_2 * width * step;
            for (unsigned i = 0; i < step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned p = 0; p < kHW; p++)
                        for (unsigned q = 0; q < kHW; q++)
                            vec[p * kHW + q + dc_p + (i * width + j) * kHW_2] =
                                        img[(p + i + 2 * nHW + 1 - step + i_r - nHW)
                                                            * width + j + q + dc];
        }

        //! Process of all DCTs
        fftwf_execute_r2r(*plan_2, vec, dct);
        fftwf_free(vec);

        //! Getting the result
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc   = c * kHW_2 * width * (2 * nHW + 1);
            const unsigned dc_p = c * kHW_2 * width * step;
            for (unsigned i = 0; i < step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned k = 0; k < kHW_2; k++)
                        DCT_table_2D[dc + ((i + 2 * nHW + 1 - step) * width + j) * kHW_2 + k] =
                            dct[dc_p + (i * width + j) * kHW_2 + k] * coef_norm[k];
        }
        fftwf_free(dct);
    }
}

/**
 * @brief Precompute a 2D bior1.5 transform on all patches contained in
 *        a part of the image.
 *
 * @param bior_table_2D : will contain the 2d bior1.5 transform for all
 *        chosen patches;
 * @param img : image on which the 2d transform will be processed;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW). MUST BE A POWER OF 2 !!!
 * @param i_r: current index of the reference patches;
 * @param step: space in pixels between two references patches;
 * @param i_min (resp. i_max) : minimum (resp. maximum) value
 *        for i_r. In this case the whole 2d transform is applied
 *        on every patches. Otherwise the precomputed 2d DCT is re-used
 *        without processing it;
 * @param lpd : low pass filter of the forward bior1.5 2d transform;
 * @param hpd : high pass filter of the forward bior1.5 2d transform.
 **/
void bior_2d_process(
    vector<float> &bior_table_2D
,   vector<float> const& img
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned step
,   const unsigned i_min
,   const unsigned i_max
,   vector<float> &lpd
,   vector<float> &hpd
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;

    //! If i_r == ns, then we have to process all Bior1.5 transforms
    if (i_r == i_min || i_r == i_max)
    {
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc = c * width * height;
            const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
            for (unsigned i = 0; i < 2 * nHW + 1; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                {
                    bior_2d_forward(img, bior_table_2D, kHW, dc +
                             (i_r + i - nHW) * width + j, width,
                             dc_p + (i * width + j) * kHW_2, lpd, hpd);
                }
        }
    }
    else
    {
        const unsigned ds = step * width * kHW_2;

        //! Re-use of Bior1.5 already processed
        for (unsigned c = 0; c < chnls; c++)
        {
            unsigned dc = c * width * (2 * nHW + 1) * kHW_2;
            for (unsigned i = 0; i < 2 * nHW + 1 - step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                    for (unsigned k = 0; k < kHW_2; k++)
                        bior_table_2D[k + (i * width + j) * kHW_2 + dc] =
                            bior_table_2D[k + (i * width + j) * kHW_2 + dc + ds];
        }

        //! Compute the new Bior
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc   = c * width * height;
            const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
            for (unsigned i = 0; i < step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                {
                    bior_2d_forward(img, bior_table_2D, kHW, dc +
                             (i + 2 * nHW + 1 - step + i_r - nHW) * width + j,
                             width, dc_p + ((i + 2 * nHW + 1 - step)
                             * width + j) * kHW_2, lpd, hpd);
                }
        }
    }
}

/**
 * @brief HT filtering using Welsh-Hadamard transform (do only third
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_3D : contains the 3D block for a reference patch;
 * @param tmp: allocated vector used in Hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param kHW : size of patches (kHW x kHW);
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param lambdaHard3D : value of thresholding;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param doWeight: if true process the weighting, do nothing
 *        otherwise.
 *
 * @return none.
 **/
void ht_filtering_hadamard(
    vector<float> &group_3D
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned kHard
,   const unsigned chnls
,   vector<float> const& sigma_table
,   const float lambdaHard3D
,   vector<float> &weight_table
,   const bool doWeight
){
    //! Declarations
    const unsigned kHard_2 = kHard * kHard;
    for (unsigned c = 0; c < chnls; c++)
        weight_table[c] = 0.0f;
    const float coef_norm = sqrtf((float) nSx_r);
    const float coef = 1.0f / (float) nSx_r;

    //! Process the Welsh-Hadamard transform on the 3rd dimension
    for (unsigned n = 0; n < kHard_2 * chnls; n++)
        hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

    //! Hard Thresholding
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * kHard_2;
        const float T = lambdaHard3D * sigma_table[c] * coef_norm;
        for (unsigned k = 0; k < kHard_2 * nSx_r; k++)
        {
            if (fabs(group_3D[k + dc]) > T)
                weight_table[c]++;
            else
                group_3D[k + dc] = 0.0f;
        }
    }

    //! Process of the Welsh-Hadamard inverse transform
    for (unsigned n = 0; n < kHard_2 * chnls; n++)
        hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

    for (unsigned k = 0; k < group_3D.size(); k++)
        group_3D[k] *= coef;

    //! Weight for aggregation
    if (doWeight)
        for (unsigned c = 0; c < chnls; c++)
            weight_table[c] = (weight_table[c] > 0.0f ? 1.0f / (float)
                    (sigma_table[c] * sigma_table[c] * weight_table[c]) : 1.0f);
}

/**
 * @brief Wiener filtering using Hadamard transform.
 *
 * @param group_3D_img : contains the 3D block built on img_noisy;
 * @param group_3D_est : contains the 3D block built on img_basic;
 * @param tmp: allocated vector used in hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param kWien : size of patches (kWien x kWien);
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param doWeight: if true process the weighting, do nothing
 *        otherwise.
 *
 * @return none.
 **/
void wiener_filtering_hadamard(
    vector<float> &group_3D_img
,   vector<float> &group_3D_est
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned kWien
,   const unsigned chnls
,   vector<float> const& sigma_table
,   vector<float> &weight_table
,   const bool doWeight
){
    //! Declarations
    const unsigned kWien_2 = kWien * kWien;
    const float coef = 1.0f / (float) nSx_r;

    for (unsigned c = 0; c < chnls; c++)
        weight_table[c] = 0.0f;

    //! Process the Welsh-Hadamard transform on the 3rd dimension
    for (unsigned n = 0; n < kWien_2 * chnls; n++)
    {
        hadamard_transform(group_3D_img, tmp, nSx_r, n * nSx_r);
        hadamard_transform(group_3D_est, tmp, nSx_r, n * nSx_r);
    }

    //! Wiener Filtering
    for (unsigned c = 0; c < chnls; c++)
    {
        const unsigned dc = c * nSx_r * kWien_2;
        for (unsigned k = 0; k < kWien_2 * nSx_r; k++)
        {
            float value = group_3D_est[dc + k] * group_3D_est[dc + k] * coef;
            value /= (value + sigma_table[c] * sigma_table[c]);
            group_3D_est[k + dc] = group_3D_img[k + dc] * value * coef;
            weight_table[c] += value;
        }
    }

    //! Process of the Welsh-Hadamard inverse transform
    for (unsigned n = 0; n < kWien_2 * chnls; n++)
        hadamard_transform(group_3D_est, tmp, nSx_r, n * nSx_r);

    //! Weight for aggregation
    if (doWeight)
        for (unsigned c = 0; c < chnls; c++)
            weight_table[c] = (weight_table[c] > 0.0f ? 1.0f / (float)
                    (sigma_table[c] * sigma_table[c] * weight_table[c]) : 1.0f);
}

/**
 * @brief Apply 2D dct inverse to a lot of patches.
 *
 * @param group_3D_table: contains a huge number of patches;
 * @param kHW : size of patch;
 * @param coef_norm_inv: contains normalization coefficients;
 * @param plan : for convenience. Used by fftw.
 *
 * @return none.
 **/
void dct_2d_inverse(
    vector<float> &group_3D_table
,   const unsigned kHW
,   const unsigned N
,   vector<float> const& coef_norm_inv
,   fftwf_plan * plan
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned size = kHW_2 * N;
	const unsigned Ns   = group_3D_table.size() / kHW_2;

    //! Allocate Memory
    float* vec = (float*) fftwf_malloc(size * sizeof(float));
    float* dct = (float*) fftwf_malloc(size * sizeof(float));

    //! Normalization
    for (unsigned n = 0; n < Ns; n++)
        for (unsigned k = 0; k < kHW_2; k++)
            dct[k + n * kHW_2] = group_3D_table[k + n * kHW_2] * coef_norm_inv[k];

    //! 2D dct inverse
    fftwf_execute_r2r(*plan, dct, vec);
    fftwf_free(dct);

    //! Getting the result + normalization
    const float coef = 1.0f / (float)(kHW * 2);
	for (unsigned k = 0; k < group_3D_table.size(); k++)
        group_3D_table[k] = coef * vec[k];

    //! Free Memory
    fftwf_free(vec);
}

void bior_2d_inverse(
    vector<float> &group_3D_table
,   const unsigned kHW
,   vector<float> const& lpr
,   vector<float> const& hpr
){
    //! Declarations
    const unsigned kHW_2 = kHW * kHW;
    const unsigned N = group_3D_table.size() / kHW_2;

    //! Bior process
    for (unsigned n = 0; n < N; n++)
        bior_2d_inverse(group_3D_table, kHW, n * kHW_2, lpr, hpr);
}

/** ----------------- **/
/** - Preprocessing - **/
/** ----------------- **/
/**
 * @brief Preprocess
 *
 * @param kaiser_window[kHW * kHW]: Will contain values of a Kaiser Window;
 * @param coef_norm: Will contain values used to normalize the 2D DCT;
 * @param coef_norm_inv: Will contain values used to normalize the 2D DCT;
 * @param bior1_5_for: will contain coefficients for the bior1.5 forward transform
 * @param bior1_5_inv: will contain coefficients for the bior1.5 inverse transform
 * @param kHW: size of patches (need to be 8 or 12).
 *
 * @return none.
 **/
void preProcess(
    vector<float> &kaiserWindow
,   vector<float> &coef_norm
,   vector<float> &coef_norm_inv
,   const unsigned kHW
){
    //! Kaiser Window coefficients
    if (kHW == 8)
    {
        //! First quarter of the matrix
        kaiserWindow[0 + kHW * 0] = 0.1924f; kaiserWindow[0 + kHW * 1] = 0.2989f; kaiserWindow[0 + kHW * 2] = 0.3846f; kaiserWindow[0 + kHW * 3] = 0.4325f;
        kaiserWindow[1 + kHW * 0] = 0.2989f; kaiserWindow[1 + kHW * 1] = 0.4642f; kaiserWindow[1 + kHW * 2] = 0.5974f; kaiserWindow[1 + kHW * 3] = 0.6717f;
        kaiserWindow[2 + kHW * 0] = 0.3846f; kaiserWindow[2 + kHW * 1] = 0.5974f; kaiserWindow[2 + kHW * 2] = 0.7688f; kaiserWindow[2 + kHW * 3] = 0.8644f;
        kaiserWindow[3 + kHW * 0] = 0.4325f; kaiserWindow[3 + kHW * 1] = 0.6717f; kaiserWindow[3 + kHW * 2] = 0.8644f; kaiserWindow[3 + kHW * 3] = 0.9718f;

        //! Completing the rest of the matrix by symmetry
        for(unsigned i = 0; i < kHW / 2; i++)
            for (unsigned j = kHW / 2; j < kHW; j++)
                kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)];

        for (unsigned i = kHW / 2; i < kHW; i++)
            for (unsigned j = 0; j < kHW; j++)
                kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j];
    }
    else if (kHW == 12)
    {
        //! First quarter of the matrix
        kaiserWindow[0 + kHW * 0] = 0.1924f; kaiserWindow[0 + kHW * 1] = 0.2615f; kaiserWindow[0 + kHW * 2] = 0.3251f; kaiserWindow[0 + kHW * 3] = 0.3782f;  kaiserWindow[0 + kHW * 4] = 0.4163f;  kaiserWindow[0 + kHW * 5] = 0.4362f;
        kaiserWindow[1 + kHW * 0] = 0.2615f; kaiserWindow[1 + kHW * 1] = 0.3554f; kaiserWindow[1 + kHW * 2] = 0.4419f; kaiserWindow[1 + kHW * 3] = 0.5139f;  kaiserWindow[1 + kHW * 4] = 0.5657f;  kaiserWindow[1 + kHW * 5] = 0.5927f;
        kaiserWindow[2 + kHW * 0] = 0.3251f; kaiserWindow[2 + kHW * 1] = 0.4419f; kaiserWindow[2 + kHW * 2] = 0.5494f; kaiserWindow[2 + kHW * 3] = 0.6390f;  kaiserWindow[2 + kHW * 4] = 0.7033f;  kaiserWindow[2 + kHW * 5] = 0.7369f;
        kaiserWindow[3 + kHW * 0] = 0.3782f; kaiserWindow[3 + kHW * 1] = 0.5139f; kaiserWindow[3 + kHW * 2] = 0.6390f; kaiserWindow[3 + kHW * 3] = 0.7433f;  kaiserWindow[3 + kHW * 4] = 0.8181f;  kaiserWindow[3 + kHW * 5] = 0.8572f;
        kaiserWindow[4 + kHW * 0] = 0.4163f; kaiserWindow[4 + kHW * 1] = 0.5657f; kaiserWindow[4 + kHW * 2] = 0.7033f; kaiserWindow[4 + kHW * 3] = 0.8181f;  kaiserWindow[4 + kHW * 4] = 0.9005f;  kaiserWindow[4 + kHW * 5] = 0.9435f;
        kaiserWindow[5 + kHW * 0] = 0.4362f; kaiserWindow[5 + kHW * 1] = 0.5927f; kaiserWindow[5 + kHW * 2] = 0.7369f; kaiserWindow[5 + kHW * 3] = 0.8572f;  kaiserWindow[5 + kHW * 4] = 0.9435f;  kaiserWindow[5 + kHW * 5] = 0.9885f;

        //! Completing the rest of the matrix by symmetry
        for(unsigned i = 0; i < kHW / 2; i++)
            for (unsigned j = kHW / 2; j < kHW; j++)
                kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)];

        for (unsigned i = kHW / 2; i < kHW; i++)
            for (unsigned j = 0; j < kHW; j++)
                kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j];
    }
    else
        for (unsigned k = 0; k < kHW * kHW; k++)
            kaiserWindow[k] = 1.0f;

    //! Coefficient of normalization for DCT II and DCT II inverse
    const float coef = 0.5f / ((float) (kHW));
    for (unsigned i = 0; i < kHW; i++)
        for (unsigned j = 0; j < kHW; j++)
        {
            if (i == 0 && j == 0)
            {
                coef_norm    [i * kHW + j] = 0.5f * coef;
                coef_norm_inv[i * kHW + j] = 2.0f;
            }
            else if (i * j == 0)
            {
                coef_norm    [i * kHW + j] = SQRT2_INV * coef;
                coef_norm_inv[i * kHW + j] = SQRT2;
            }
            else
            {
                coef_norm    [i * kHW + j] = 1.0f * coef;
                coef_norm_inv[i * kHW + j] = 1.0f;
            }
        }
}

/**
 * @brief Precompute Bloc Matching (distance inter-patches)
 *
 * @param patch_table: for each patch in the image, will contain
 * all coordonnate of its similar patches
 * @param img: noisy image on which the distance is computed
 * @param width, height: size of img
 * @param kHW: size of patch
 * @param NHW: maximum similar patches wanted
 * @param nHW: size of the boundary of img
 * @param tauMatch: threshold used to determinate similarity between
 *        patches
 *
 * @return none.
 **/
void precompute_BM(
    vector<vector<unsigned> > &patch_table
,   const vector<float> &img
,   const unsigned width
,   const unsigned height
,   const unsigned kHW
,   const unsigned NHW
,   const unsigned nHW
,   const unsigned pHW
,   const float    tauMatch
){
    //! Declarations
    const unsigned Ns = 2 * nHW + 1;
    const float threshold = tauMatch * kHW * kHW;
    vector<float> diff_table(width * height);
    vector<vector<float> > sum_table((nHW + 1) * Ns, vector<float> (width * height, 2 * threshold));
    if (patch_table.size() != width * height)
        patch_table.resize(width * height);
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHW + 1, nHW, pHW);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHW + 1, nHW, pHW);

    //! For each possible distance, precompute inter-patches distance
    for (unsigned di = 0; di <= nHW; di++)
        for (unsigned dj = 0; dj < Ns; dj++)
        {
            const int dk = (int) (di * width + dj) - (int) nHW;
            const unsigned ddk = di * Ns + dj;

            //! Process the image containing the square distance between pixels
            for (unsigned i = nHW; i < height - nHW; i++)
            {
                unsigned k = i * width + nHW;
                for (unsigned j = nHW; j < width - nHW; j++, k++)
                    diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k]);
            }

            //! Compute the sum for each patches, using the method of the integral images
            const unsigned dn = nHW * width + nHW;
            //! 1st patch, top left corner
            float value = 0.0f;
            for (unsigned p = 0; p < kHW; p++)
            {
                unsigned pq = p * width + dn;
                for (unsigned q = 0; q < kHW; q++, pq++)
                    value += diff_table[pq];
            }
            sum_table[ddk][dn] = value;

            //! 1st row, top
            for (unsigned j = nHW + 1; j < width - nHW; j++)
            {
                const unsigned ind = nHW * width + j - 1;
                float sum = sum_table[ddk][ind];
                for (unsigned p = 0; p < kHW; p++)
                    sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width];
                sum_table[ddk][ind + 1] = sum;
            }

            //! General case
            for (unsigned i = nHW + 1; i < height - nHW; i++)
            {
                const unsigned ind = (i - 1) * width + nHW;
                float sum = sum_table[ddk][ind];
                //! 1st column, left
                for (unsigned q = 0; q < kHW; q++)
                    sum += diff_table[ind + kHW * width + q] - diff_table[ind + q];
                sum_table[ddk][ind + width] = sum;

                //! Other columns
                unsigned k = i * width + nHW + 1;
                unsigned pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1;
                for (unsigned j = nHW + 1; j < width - nHW; j++, k++, pq++)
                {
                    sum_table[ddk][k] =
                          sum_table[ddk][k - 1]
                        + sum_table[ddk][k - width]
                        - sum_table[ddk][k - 1 - width]
                        + diff_table[pq]
                        - diff_table[pq - kHW]
                        - diff_table[pq - kHW * width]
                        + diff_table[pq - kHW - kHW * width];
                }

            }
        }

    //! Precompute Bloc Matching
    vector<pair<float, unsigned> > table_distance;
    //! To avoid reallocation
    table_distance.reserve(Ns * Ns);

    for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
    {
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            //! Initialization
            const unsigned k_r = row_ind[ind_i] * width + column_ind[ind_j];
            table_distance.clear();
            patch_table[k_r].clear();

            //! Threshold distances in order to keep similar patches
            for (int dj = -(int) nHW; dj <= (int) nHW; dj++)
            {
                for (int di = 0; di <= (int) nHW; di++)
                    if (sum_table[dj + nHW + di * Ns][k_r] < threshold)
                        table_distance.push_back(make_pair(
                                    sum_table[dj + nHW + di * Ns][k_r]
                                  , k_r + di * width + dj));

                for (int di = - (int) nHW; di < 0; di++)
                    if (sum_table[-dj + nHW + (-di) * Ns][k_r] < threshold)
                        table_distance.push_back(make_pair(
                                    sum_table[-dj + nHW + (-di) * Ns][k_r + di * width + dj]
                                  , k_r + di * width + dj));
            }

            //! We need a power of 2 for the number of similar patches,
            //! because of the Welsh-Hadamard transform on the third dimension.
            //! We assume that NHW is already a power of 2
            const unsigned nSx_r = (NHW > table_distance.size() ?
                                    closest_power_of_2(table_distance.size()) : NHW);

			//! To avoid problem
			if (nSx_r == 1 && table_distance.size() == 0)
			{
				cout << "problem size" << endl;
				table_distance.push_back(make_pair(0, k_r));
			}

            //! Sort patches according to their distance to the reference one
            partial_sort(table_distance.begin(), table_distance.begin() + nSx_r,
                                            table_distance.end(), ComparaisonFirst);

            //! Keep a maximum of NHW similar patches
            for (unsigned n = 0; n < nSx_r; n++)
                patch_table[k_r].push_back(table_distance[n].second);

			//! To avoid problem
			if (nSx_r == 1)
				patch_table[k_r].push_back(table_distance[0].second);
        }
    }
}

/**
 * @brief Process of a weight dependent on the standard
 *        deviation, used during the weighted aggregation.
 *
 * @param group_3D : 3D group
 * @param nSx_r : number of similar patches in the 3D group
 * @param kHW: size of patches
 * @param chnls: number of channels in the image
 * @param weight_table: will contain the weighting for each
 *        channel.
 *
 * @return none.
 **/
void sd_weighting(
    std::vector<float> const& group_3D
,   const unsigned nSx_r
,   const unsigned kHW
,   const unsigned chnls
,   std::vector<float> &weight_table
){
    const unsigned N = nSx_r * kHW * kHW;

    for (unsigned c = 0; c < chnls; c++)
    {
        //! Initialization
        float mean = 0.0f;
        float std  = 0.0f;

        //! Compute the sum and the square sum
        for (unsigned k = 0; k < N; k++)
        {
            mean += group_3D[k];
            std  += group_3D[k] * group_3D[k];
        }

        //! Sample standard deviation (Bessel's correction)
        float res = (std - mean * mean / (float) N) / (float) (N - 1);

        //! Return the weight as used in the aggregation
        weight_table[c] = (res > 0.0f ? 1.0f / sqrtf(res) : 0.0f);
    }
}






