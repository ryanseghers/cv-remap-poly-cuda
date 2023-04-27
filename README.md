# C++ OpenCV Image Remap by 2-d Cubic Polynomial with CUDA

## Introduction
This implements a 2-d cubic polynomial image warp/remap operation in plain C++, AVX intrinsics, and in CUDA to investigate performance of this approach against the OpenCV approach of specifying the warp/remap via two float32 images.

## Background
An image warping operation can be described by specifying, for each pixel in the output image, the corresponding location to sample in the source image. This is how the OpenCV remap operation works. It uses two float32 images to store those source points; one image for the x values and one for the y values. Note that they are floating point values and so specify subpixel locations and various sampling algorithms can be used for example nearest-neighbor, bilinear, and bicubic.

Optical distortions such as pincushion and barrel can be described by a field of (small) vectors (dx, dy) where each vector specifies the offset applied to a pixel to move it from distorted to undistorted location (or vice versa). For certain distortions, the dx and dy values can be modeled (separately) by a 2-d third-order (cubic) polynomial:

    c0 + (c1 * x) + (c2 * y) + (c3 * x^2) + (c4 * xy) + (c5 * y^2) + (c6 * x^3) + (c7 * x^2*y) + (c8 * y^2*x) + (c9 * y^3)

(where x,y is the output image pixel location and c0 to c9 are the polynomial coefficients)

Thus the optical distortion model comprises two sets of coefficients for that polynomial: one for the field of dx values and one for the field of dy values. Thus for each output image pixel, one can compute the offset to the input location to sample by evaluating the polynomial once with the dx coefficients and once with the dy coefficients.

## Purpose
The OpenCV approach of representing sample points with two float images is probably best in most scenarios, however note that two 32-bit float images (which have same size as the output image) consume some memory and some memory bandwidth. Either of those may be an issue in some circumstances, for example in the GPGPU scenario when there is limited memory on the device and/or limited memory bandwidth between host and device. Using the 2-d polynomial approach, instead of copying two full-sized float32 images to the device one just needs to pass the 20 polynomial coefficients. For that reason I was interested to implement this polynomial-based remap and investigate the performance in the GPGPU scenario.

## Results
I've written a canonical C++ implementation, an x64 intrinsics AVX implementation, and the CUDA implementation and some benchmark functions to evaluate performance.

TODO: add benchmark section
