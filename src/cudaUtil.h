#pragma once
#include "cuda_runtime.h"

void assertCudaStatus(cudaError_t cudaStatus, const char* msg);
void setDevice(int deviceId = 0);
void printDeviceInfo(int deviceId = 0);
void setupSrcImageTexture16u(int width, int height, const uint16_t* psrc, int samplingType, cudaArray_t& srcArray, cudaTextureObject_t& srcTexture);
