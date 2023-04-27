#pragma once
#include <immintrin.h>
#include "poly.h"

namespace CVRemap
{
    __m256 evalPolyAvx(float* coeffs, __m256 x_vals, __m256 y_vals);
    void buildRemapImagesAvx(int width, int height, RemapCoeffs3rdOrder& coeffs, cv::Mat& dxImg, cv::Mat& dyImg);
    void remapPoly16uAvx(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, int samplingType, cv::Mat& dst);
}
