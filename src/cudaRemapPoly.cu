#include <array>
#include <cstdio>
#include <exception>
#include <cuda_runtime.h>

#include "device_launch_parameters.h"
#include "cudaRemapPoly.h"
#include "cudaUtil.h"

/**
 * @brief Eval the cubic polynomial.
*/
__device__ float evalPoly(int x, int y, float* coeffs)
{
    // 1, x, y, x^2, xy, y^2, x^3, x^2*y, y^2*x, y^3
    float x2 = x * x;
    float y2 = y * y;

    return coeffs[0]
        + coeffs[1] * x
        + coeffs[2] * y
        + coeffs[3] * x2
        + coeffs[4] * x * y
        + coeffs[5] * y2
        + coeffs[6] * x2 * x
        + coeffs[7] * x2 * y
        + coeffs[8] * y2 * x
        + coeffs[9] * y2 * y;
}

/**
 * @brief Single cubic interpolation at location t between four values.
*/
__device__ float cubicInterpolate(float p0, float p1, float p2, float p3, float t)
{
    float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    float a2 = -0.5f * p0 + 0.5f * p2;
    float a3 = p1;

    float t2 = t * t;
    return a0 * t2 * t + a1 * t2 + a2 * t + a3;
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

/**
 * @brief Sample a single point from the specified source texture using bicubic interpolation.
*/
__device__ float bicubicSample(cudaTextureObject_t srcTexture, int width, int height, float x, float y)
{
    int x1 = clamp(static_cast<int>(floor(x)) - 1, 0, width - 1);
    int y1 = clamp(static_cast<int>(floor(y)) - 1, 0, height - 1);
    int x2 = clamp(x1 + 1, 0, width - 1);
    int y2 = clamp(y1 + 1, 0, height - 1);
    int x3 = clamp(x1 + 2, 0, width - 1);
    int y3 = clamp(y1 + 2, 0, height - 1);
    int x4 = clamp(x1 + 3, 0, width - 1);
    int y4 = clamp(y1 + 3, 0, height - 1);

    float dx = x - x2;
    float dy = y - y2;

    float row_interpolations[4];

    for (int i = 0; i < 4; ++i)
    {
        float p1 = (float)tex2D<uint16_t>(srcTexture, x1, y1 + i);
        float p2 = (float)tex2D<uint16_t>(srcTexture, x2, y1 + i);
        float p3 = (float)tex2D<uint16_t>(srcTexture, x3, y1 + i);
        float p4 = (float)tex2D<uint16_t>(srcTexture, x4, y1 + i);

        row_interpolations[i] = cubicInterpolate(p1, p2, p3, p4, dx);
    }

    return cubicInterpolate(row_interpolations[0], row_interpolations[1], row_interpolations[2], row_interpolations[3], dy);
}

/**
 * @brief Kernel to apply the poly to remap a pixel.
 * Note that samplingType is the same for all threads so warp can run efficiently.
 * @param outputData Output image.
 * @param width Input and output image width, in pixels.
 * @param height Input and output image height, in pixels.
 * @param dxCoeffs Poly coeffs.
 * @param dyCoeffs Poly coeffs.
 * @param samplingType 0=NN, 1=bilinear, 2=bicubic
 * @param srcTexture 
*/
__global__ void remapKernel(uint16_t* outputData, int width, int height, float* dxCoeffs, float* dyCoeffs, int samplingType, cudaTextureObject_t srcTexture)
{
    // which output pixel this thread is for
    unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;

    if ((yi >= height) || (xi >= width))
    {
        return;
    }

    // where to sample from source image (texture)
    float x = xi + evalPoly(xi, yi, dxCoeffs);
    float y = yi + evalPoly(xi, yi, dyCoeffs);

    if (samplingType == 0)
    {
        // nearest neighbor
        outputData[yi * width + xi] = tex2D<uint16_t>(srcTexture, x + 0.5f, y + 0.5f);
    }
    else if (samplingType == 1)
    {
        // bilinear, returns float
        float val = tex2D<float>(srcTexture, x + 0.5f, y + 0.5f);
        outputData[yi * width + xi] = (uint16_t)(val * 65535.0f);
    }
    else if (samplingType == 2)
    {
        // bicubic
        // need to clamp to border to avoid artifacts
        const int borderPx = 2;

        if ((x >= borderPx) && (x < width - borderPx) && (y >= borderPx) && (y < height - borderPx))
        {
            outputData[yi * width + xi] = (uint16_t)(bicubicSample(srcTexture, width, height, x, y) + 0.5f);
        }
        else
        {
            outputData[yi * width + xi] = (uint16_t)0;
        }
    }
}

/**
 * @brief Host CUDA remap poly entrypoint.
 * Even bicubic is 85% memory throughput (texture cache) limited.
 * @param deviceId 
 * @param width Input and output image width, in pixels.
 * @param height Input and output image height, in pixels.
 * @param psrc Source image.
 * @param dxCoeffs Poly coeffs.
 * @param dyCoeffs Poly coeffs.
 * @param samplingType 0=NN, 1=bilinear, 2=bicubic
 * @param pdst Destination image.
*/
void cudaRemapPoly(int deviceId, int width, int height, const uint16_t* psrc, const std::array<float, 10>& dxCoeffs,
    const std::array<float, 10>& dyCoeffs, int samplingType, uint16_t* pdst)
{
    cudaError_t cudaStatus;

    // setting device multiple times has no impact
    setDevice(deviceId);

    size_t imageLen = width * height * sizeof(uint16_t);

    uint16_t* pdst_dev = nullptr;
    float* dx_coeffs_dev = nullptr;
    float* dy_coeffs_dev = nullptr;

    cudaArray_t srcArray = {};
    cudaTextureObject_t srcTexture = {};
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint16_t>();

    // IIRC there are 128 cuda cores per SM, so maybe 128 threads per thread block is good?
    // yes: changing to 16x8=128 did increase theoretical occupancy to 100% (but didn't actually speed this up)
    // apparently perf is insensitive to 4x4 and 16x16 because bottleneck is texture cache
    dim3 dimBlock(16, 8, 1);

    // however many thread blocks to cover image
    // (if image is not multiple of thread block dims then some will not be computed)
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1); 

    // dst image
    if (cudaMalloc((void**)&pdst_dev, imageLen) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dest image failed!");
        goto Error;
    }

    // coeffs
    if (cudaMalloc((void**)&dx_coeffs_dev, dxCoeffs.size() * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dx coeffs failed!");
        goto Error;
    }

    if (cudaMalloc((void**)&dy_coeffs_dev, dyCoeffs.size() * sizeof(float)) != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dy coeffs failed!");
        goto Error;
    }

    // copy coeffs
    if (cudaMemcpy(dx_coeffs_dev, dxCoeffs.data(), dxCoeffs.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy dx coeffs failed!");
        goto Error;
    }

    if (cudaMemcpy(dy_coeffs_dev, dyCoeffs.data(), dyCoeffs.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy dy coeffs failed!");
        goto Error;
    }

    // src image
    setupSrcImageTexture16u(width, height, psrc, samplingType, srcArray, srcTexture);

    // kernel
    // clang-format off
    remapKernel<<<dimGrid, dimBlock, 0>>>(pdst_dev, width, height, dx_coeffs_dev, dy_coeffs_dev, samplingType, srcTexture);
    // clang-format on
 
    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // wait for the kernel to finish and check errors
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // output
    if (cudaMemcpy(pdst, pdst_dev, imageLen, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy dest image failed!");
        goto Error;
    }

Error:
    cudaDestroyTextureObject(srcTexture);
    cudaFreeArray(srcArray);

    cudaFree(pdst_dev);
    cudaFree(dx_coeffs_dev);
    cudaFree(dy_coeffs_dev);
}
