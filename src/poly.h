#pragma once
#include <array>
#include <opencv2/opencv.hpp>

namespace CVRemap
{
    /**
    * @brief Remap polynomial coefficients.
    * The coeff order is:
    *     1, x, y, x^2, xy, y^2, x^3, x^2*y, y^2*x, y^3
    * @see http://air.bmap.ucla.edu/AIR5/2Dnonlinear.html
    * For function:
    *   f(x,y) = c0 + (c1 * x) + (c2 * y) + (c3 * x^2) + (c4 * xy) + (c5 * y^2) + (c6 * x^3) + (c7 * x^2*y) + (c8 * y^2*x) + (c9 * y^3)
    */
    struct RemapCoeffs3rdOrder
    {
        std::array<float, 10> dxCoeffs;
        std::array<float, 10> dyCoeffs;

        static float evalPoly(std::array<float, 10>& coeffs, float x, float y);

        float evalDx(float x, float y);
        float evalDy(float x, float y);
    };
}