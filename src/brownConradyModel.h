#pragma once
#include <opencv2/opencv.hpp>

namespace CVRemap
{
    /**
    * @brief Coefficients for Brown-Conrady model for lens distortion.
    * This includes radial and tangential coefficients.
    */
    struct BrownConradyCoeffs
    {
        std::array<float, 6> radial;
        float p1; // tangential
        float p2; // tangential

        BrownConradyCoeffs() : radial(), p1(0.0f), p2(0.0f)
        {
        }
    };

    void buildBrownConradyRemapImages(int width, int height, const BrownConradyCoeffs& coeffs, cv::Mat& xMap, cv::Mat& yMap);
    cv::Point2f applyBrownConradyModel(int width, int height, int xi, int yi, const BrownConradyCoeffs& coeffs);
    void applyBrownConradyModel(cv::Mat& src, const BrownConradyCoeffs& coeffs, int samplingType, cv::Mat& dst);
}