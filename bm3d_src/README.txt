% BM3D image denoising.

# ABOUT

* Author    : Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
* Copyright : (C) 2011 IPOL Image Processing On Line http://www.ipol.im/
* Licence   : GPL v3+, see GPLv3.txt

# OVERVIEW

This source code provides an implementation of the BM3D image denoising.

# UNIX/LINUX/MAC USER GUIDE

The code is compilable on Unix/Linux and Mac OS. 

- Compilation. 
Automated compilation requires the make program.

- Library. 
This code requires the libpng library and the fftw library.

- Image format. 
Only the PNG format is supported. 
 
-------------------------------------------------------------------------
Usage:
1. Download the code package and extract it. Go to that directory. 

2. Compile the source code (on Unix/Linux/Mac OS). 
There are two ways to compile the code. 
(1) RECOMMENDED, with Open Multi-Processing multithread parallelization 
(http://openmp.org/). Roughly speaking, it accelerates the program using the 
multiple processors in the computer. Run
make OMP=1

OR
(2) If the complier does not support OpenMp, run 
make

3. Run BM3D image denoising.
./BM3Ddenoising
The generic way to run the code is:

./BM3Ddenoising cinput.png sigma ImNoisy.png ImBasic.png ImDenoised.png ImDiff.png ImBias.png
ImDiffBias.png computeBias 2DtransformStep1 useSD1 2DtransformStep2 useSD2 ColorSpace

with :
- cinput.png is a noise-free image;
- sigma is the value of the noise which will be added to cinput.png;
- ImNoisy.png will contain the noisy image;
- ImBasic.png will contain the result of the first step of the algorithm;
- ImDenoised.png will contain the final result of the algorithm;
- ImDiff.png will contain the difference between cinput.png and ImDenoised.png;
- ImBias.png will contain the result of the algorithm applied on cinput.png;
- ImDiffBias.png will contain the difference between cinput.png and ImBias.png;
- computeBias : see (3);
- 2DtransformStep1: choice of the 2D transform which will be applied in the first step of the
algorithm. See (4);
- useSD1 : see (1) below;
- 2DtransformStep2: choice of the 2D transform which will be applied in the second step of the
algorithm. See (4);
- useSD2 : see (2);
- ColorSpace : choice of the color space on which the image will be applied. See (5).

There are multiple ways to run the code:
(1) for the first step, users can choose if they prefer to use
standard variation for the weighted aggregation (useSD1 = 1)
(2) for the second step, users can choose if they prefer to use
standard variation for the weighted aggregation (useSD2 = 1)
(3) you can moreover want to compute the bias (algorithm applied to the original
image). To do this, use computeBias = 1.
(4) you can choose the DCT transform or the Bior1.5 transform for the 2D transform
in the step 1 (tau_2D_hard = dct or bior) and/or the step 2. (tau_2d_wien = dct or
bior).
(5) you can choose the colorspace for both steps between : rgb, yuv, ycbcr and opp.
 
Example, run
./BM3Ddenoising cinput.png 10 ImNoisy.png ImBasic.png ImDenoised.png ImDiff.png ImBias.png
ImDiffBias.png 1 bior 0 dct 1 opp


# ABOUT THIS FILE

Copyright 2011 IPOL Image Processing On Line http://www.ipol.im/

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without any warranty.
