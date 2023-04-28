#include <fmt/core.h>
#include "poly.h"

namespace CVRemap
{
    /**
    * @brief Compute dx at the specified source x,y point.
    * @return The offset (on x axis) to sample from the source for this dest image location.
    */
    float RemapCoeffs3rdOrder::evalPoly(std::array<float, 10>& coeffs, float x, float y)
    {
        // 1 + x + y + x^2 + xy + y^2 + x^3 + x^2*y + y^2*x + y^3
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
    * @brief Compute dx at the specified source x,y point.
    * @return The offset (on x axis) to sample from the source for this dest image location.
    */
    float RemapCoeffs3rdOrder::evalDx(float x, float y)
    {
        return evalPoly(dxCoeffs, x, y);
    }

    /**
    * @brief Compute dy at the specified source x,y point.
    * @return The offset (on y axis) to sample from the source for this dest image location.
    */
    float RemapCoeffs3rdOrder::evalDy(float x, float y)
    {
        return evalPoly(dyCoeffs, x, y);
    }

    void RemapCoeffs3rdOrder::print()
    {
        fmt::print("X: ");

        for (int i = 0; i < dxCoeffs.size(); i++)
        {
            fmt::print("{:.3f}, ", dxCoeffs[i]);
        }

        fmt::print("\n");

        fmt::print("Y: ");

        for (int i = 0; i < dyCoeffs.size(); i++)
        {
            fmt::print("{:.3f}, ", dyCoeffs[i]);
        }

        fmt::print("\n");
    }
}
