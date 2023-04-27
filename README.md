# cv-remap-poly-cuda
C++ OpenCV Image Remap by Polynomial with CUDA

This is a work in progress, not particularly ready to be shared.

Define a 2-d polynomial to specify an image warp operation that can fit optical distortion such as pincushion. Pass those coefficients to GPU to compute the offsets to resample each pixel.

I also implemented a canonical C++ implementation, and an x64 intrinsics AVX implementation for comparison.

I implemented bilinear and bicubic sampling on CPU but not in CUDA yet.

