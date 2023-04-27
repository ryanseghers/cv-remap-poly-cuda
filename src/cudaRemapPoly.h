#pragma once
#include <array>
#include <cstdint>
#include "cuda_runtime.h"

void cudaRemapPoly(int deviceId, int width, int height, const uint16_t* psrc, const std::array<float, 10>& dxCoeffs, 
    const std::array<float, 10>& dyCoeffs, int samplingType, uint16_t* pdst);
