#include <cstdio>
#include <exception>
#include "cudaUtil.h"

void assertCudaStatus(cudaError_t cudaStatus, const char* msg)
{
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, msg);
        throw std::exception(msg);
    }
}

/**
 * @brief Setting device multiple times apparently has no impact, perf or otherwise.
*/
void setDevice(int deviceId)
{
    assertCudaStatus(cudaSetDevice(deviceId), "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
}

void printDeviceInfo(int deviceId)
{
    setDevice(deviceId);

    int major, minor;
    assertCudaStatus(cudaDeviceGetAttribute(&major, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, deviceId), "Unable to get device compute capability.");
    assertCudaStatus(cudaDeviceGetAttribute(&minor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor, deviceId), "Unable to get device compute capability.");

    int threadsPerSm;
    assertCudaStatus(cudaDeviceGetAttribute(&threadsPerSm, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, deviceId), "Unable to get device capability.");

    int smCount;
    assertCudaStatus(cudaDeviceGetAttribute(&smCount, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, deviceId), "Unable to get device capability.");

    int kernelCount;
    assertCudaStatus(cudaDeviceGetAttribute(&kernelCount, cudaDeviceAttr::cudaDevAttrConcurrentKernels, deviceId), "Unable to get device capability.");

    int clockRate;
    assertCudaStatus(cudaDeviceGetAttribute(&clockRate, cudaDeviceAttr::cudaDevAttrClockRate, deviceId), "Unable to get device capability.");

    fprintf(stderr, "Device %d compute capability: %d.%d\n", deviceId, major, minor);
    fprintf(stderr, "SM count: %d\n", smCount);
    fprintf(stderr, "Concurrent kernels count: %d\n", kernelCount);
    fprintf(stderr, "Clock rate: %d MHz\n", clockRate / 1000);
    fprintf(stderr, "Threads per SM: %d\n", threadsPerSm);
}

/**
* @brief Create a 2-d source texture for uint16 data type.
* @param tex Output.
* @param cuArray The source data (uint16) array.
* @param samplingType Affects texture filter and read modes.
*/
void createSourceTexture16u(cudaTextureObject_t& tex, cudaArray_t& cuArray, int samplingType)
{
    cudaResourceDesc resourceDesc = {};
    resourceDesc.res.array.array = cuArray;
    resourceDesc.resType = cudaResourceTypeArray;

    cudaTextureDesc textureDesc = {};
    textureDesc.normalizedCoords = false; // normalized is 0 to 1

    if (samplingType == 1)
    {
        // bilinear
        textureDesc.filterMode = cudaFilterModeLinear;
        textureDesc.readMode = cudaReadModeNormalizedFloat; // have to do this in order to use texture bilinear
    }
    else
    {
        // for nearest neighbor and bicubic we do nearest neighbor (because we will be doing bicubic ourselves)
        textureDesc.filterMode = cudaFilterModePoint;
        textureDesc.readMode = cudaReadModeElementType; // orig type (uint16) rather than float
    }

    cudaTextureAddressMode addressMode = cudaAddressModeBorder; // border means return 0 when off-image rather than clamp the coords (or wrap)
    textureDesc.addressMode[0] = addressMode; 
    textureDesc.addressMode[1] = addressMode;

    auto cudaStatus = cudaCreateTextureObject(&tex, &resourceDesc, &textureDesc, NULL);
    assertCudaStatus(cudaStatus, "cudaCreateTextureObject failed.");
}

/**
* @brief Create a cudaArray and cudaTextureObject and copy source data to device.
* You must free the cudaArray and cudaTextureObject when done.
* @param width 
* @param height 
* @param psrc Source data.
* @param samplingType Affects how the texture is set up.
* @param srcArray Output.
* @param srcTexture Output.
*/
void setupSrcImageTexture16u(int width, int height, const uint16_t* psrc, int samplingType, cudaArray_t& srcArray, cudaTextureObject_t& srcTexture)
{
    size_t imageLen = width * height * sizeof(uint16_t);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint16_t>();

    // src image
    auto cudaStatus = cudaMallocArray(&srcArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);
    assertCudaStatus(cudaStatus, "CUDA malloc array failed.");
    cudaStatus = cudaMemcpyToArray(srcArray, 0, 0, psrc, imageLen, cudaMemcpyHostToDevice);
    assertCudaStatus(cudaStatus, "CUDA copy src image to array failed.");
    createSourceTexture16u(srcTexture, srcArray, samplingType);
}
