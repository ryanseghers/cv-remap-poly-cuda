#include "CVRemap.h"

namespace CVRemap
{
    inline float cubicInterpolate(float p0, float p1, float p2, float p3, float t)
    {
        float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
        float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
        float a2 = -0.5f * p0 + 0.5f * p2;
        float a3 = p1;

        float t2 = t * t;
        return a0 * t2 * t + a1 * t2 + a2 * t + a3;
    }

    void remapPoly(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, int samplingType, cv::Mat& dst)
    {
        dst.create(src.rows, src.cols, src.type());

        if (samplingType == cv::INTER_NEAREST)
        {
            remapPolyNearest<uint16_t>(src, coeffs, dst);
        }
        else if (samplingType == cv::INTER_LINEAR)
        {
            remapPolyBilinear<uint16_t>(src, coeffs, dst);
        }
        else if (samplingType == cv::INTER_CUBIC)
        {
            remapPolyBicubic<uint16_t>(src, coeffs, dst);
        }
        else
        {
            throw std::runtime_error("Unimplemented sampling type.");
        }
    }
}
