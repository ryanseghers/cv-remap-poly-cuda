#include "poly.h"
#include "avxRemapPoly.h"

namespace CVRemap
{
    /**
    * @brief Eval the poly for 8 x,y points.
    * f(x,y) = c0 + (c1 * x) + (c2 * y) + (c3 * x^2) + (c4 * xy) + (c5 * y^2) + (c6 * x^3) + (c7 * x^2*y) + (c8 * y^2*x) + (c9 * y^3)
    */
    __m256 evalPolyAvx(float* coeffs, __m256 x_vals, __m256 y_vals)
    {
        __m256 c0 = _mm256_set1_ps(coeffs[0]);
        __m256 sum = c0;

        // re-used values
        __m256 x2 = _mm256_mul_ps(x_vals, x_vals);
        __m256 xy = _mm256_mul_ps(x_vals, y_vals);
        __m256 y2 = _mm256_mul_ps(y_vals, y_vals);

        __m256 c1 = _mm256_set1_ps(coeffs[1]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(c1, x_vals));        // + c1 * x

        __m256 c2 = _mm256_set1_ps(coeffs[2]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(c2, y_vals));        // + c2 * y

        __m256 c3 = _mm256_set1_ps(coeffs[3]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(c3, x2)); // + c3 * x * x

        __m256 c4 = _mm256_set1_ps(coeffs[4]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(c4, xy)); // + c4 * x * y

        __m256 c5 = _mm256_set1_ps(coeffs[5]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(c5, y2)); // + c5 * y * y

        __m256 c6 = _mm256_set1_ps(coeffs[6]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(c6, _mm256_mul_ps(x_vals, x2))); // + c6 * x * x * x

        __m256 c7 = _mm256_set1_ps(coeffs[7]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(c7, _mm256_mul_ps(x2, y_vals))); // + c7 * x * x * y

        __m256 c8 = _mm256_set1_ps(coeffs[8]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(c8, _mm256_mul_ps(x_vals, y2))); // + c8 * x * y * y

        __m256 c9 = _mm256_set1_ps(coeffs[9]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(c9, _mm256_mul_ps(y_vals, y2))); // + c9 * y * y * y

        return sum;
    }

    /**
     * @brief Use AVX eval poly to fill in dx and dy images for OpenCV remap.
     * @param width Image width.
     * @param height Image height.
     * @param coeffs Poly coeffs.
     * @param dxImg Output.
     * @param dyImg Output.
    */
    void buildRemapImagesAvx(int width, int height, RemapCoeffs3rdOrder& coeffs, cv::Mat& dxImg, cv::Mat& dyImg)
    {
        dxImg.create(height, width, CV_32F);
        dyImg.create(height, width, CV_32F);

        // AVX registers are 256 bits (8 floats)
        int nBlocks = width / 8;
        __m256 x_coeff = _mm256_loadu_ps(coeffs.dxCoeffs.data());
        __m256 y_coeff = _mm256_loadu_ps(coeffs.dyCoeffs.data());

        // constants to quickly add to create the x values for each of the 8 pixels we process at once
        __m256 x_indices = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);

        for (int y = 0; y < height; y++)
        {
            __m256 y_vals = _mm256_set1_ps((float)y);

            for (int xi = 0; xi < nBlocks; xi++)
            {
                int x = xi * 8;
                __m256 x_vals = _mm256_set1_ps((float)x);
                x_vals = _mm256_add_ps(x_vals, x_indices);

                // dx
                __m256 dx_vals = evalPolyAvx(coeffs.dxCoeffs.data(), x_vals, y_vals);
                __m256 new_x_vals = _mm256_add_ps(x_vals, dx_vals);
                _mm256_storeu_ps(dxImg.ptr<float>(y, x), new_x_vals);

                // dy
                __m256 dy_vals = evalPolyAvx(coeffs.dyCoeffs.data(), x_vals, y_vals);
                __m256 new_y_vals = _mm256_add_ps(y_vals, dy_vals);
                _mm256_storeu_ps(dyImg.ptr<float>(y, x), new_y_vals);
            }

            // tail
            for (int x = nBlocks * 8; x < width; x++)
            {
                dxImg.at<float>(y, x) = x + coeffs.evalDx(x, y);
                dyImg.at<float>(y, x) = y + coeffs.evalDy(x, y);
            }
        }
    }

    /**
     * @brief Use AVX eval poly to do nearest-neighbor remap.
     * I tried a variation that clamped the x and y coordinates to acceptable ranges to avoid the branch but it was slower.
    */
    void remapPoly16uAvxNearest(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, cv::Mat& dst)
    {
        int width = src.cols;
        int height = src.rows;

        // AVX registers are 256 bits (8 floats)
        int nBlocks = width / 8;
        __m256 x_coeff = _mm256_loadu_ps(coeffs.dxCoeffs.data());
        __m256 y_coeff = _mm256_loadu_ps(coeffs.dyCoeffs.data());

        // constants to quickly add to create the x values for each of the 8 pixels we process at once
        __m256 x_indices = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);

        for (int y = 0; y < height; y++)
        {
            __m256 y_vals = _mm256_set1_ps((float)y);

            for (int xi = 0; xi < nBlocks; xi++)
            {
                int x = xi * 8;
                __m256 x_vals = _mm256_set1_ps((float)x);
                x_vals = _mm256_add_ps(x_vals, x_indices);

                // dx
                __m256 dx_vals = evalPolyAvx(coeffs.dxCoeffs.data(), x_vals, y_vals);
                __m256 new_x_vals = _mm256_add_ps(x_vals, dx_vals);
                __m256i x_ints = _mm256_cvtps_epi32(new_x_vals);

                // dy
                __m256 dy_vals = evalPolyAvx(coeffs.dyCoeffs.data(), x_vals, y_vals);
                __m256 new_y_vals = _mm256_add_ps(y_vals, dy_vals);
                __m256i y_ints = _mm256_cvtps_epi32(new_y_vals);

                // there's no gather for 16u
                for (int i = 0; i < 8; i++)
                {
                    uint16_t val = 0;
                    int x_src = x_ints.m256i_i32[i];
                    int y_src = y_ints.m256i_i32[i];

                    if ((x_src >= 0) && (x_src < src.cols) && (y_src >= 0) && (y_src < src.rows))
                    {
                        val = src.at<uint16_t>(y_src, x_src);
                    }

                    dst.at<uint16_t>(y, x + i) = val;
                }
            }

            // tail
            for (int x = nBlocks * 8; x < width; x++)
            {
                float dx = coeffs.evalDx(x, y);
                float dy = coeffs.evalDy(x, y);

                int x_src = (int)(x + dx + 0.5f);
                int y_src = (int)(y + dy + 0.5f);

                uint16_t val = 0;

                if ((x_src >= 0) && (x_src < src.cols) && (y_src >= 0) && (y_src < src.rows))
                {
                    val = src.at<uint16_t>(y_src, x_src);
                }

                dst.at<uint16_t>(y, x) = val;
            }
        }
    }

    /**
     * @brief Remap poly a 16u image given the coeffs.
     * This only implements nearest neighbor so far.
    */
    void remapPoly16uAvx(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, int samplingType, cv::Mat& dst)
    {
        dst.create(src.rows, src.cols, src.type());

        if (samplingType == cv::INTER_NEAREST)
        {
            remapPoly16uAvxNearest(src, coeffs, dst);
        }
        else
        {
            throw std::runtime_error("Unimplemented sampling type.");
        }
    }
}
