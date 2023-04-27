#pragma once
#include <array>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

#include "poly.h"

namespace CVRemap
{
    void remapPoly(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, int samplingType, cv::Mat& dst);

    /**
     * @brief Nearest-neighbor.
    */
    template <typename T>
    void remapPolyNearest(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, cv::Mat& dst)
    {
        dst.create(src.rows, src.cols, src.type());

        for (int y = 0; y < src.rows; y++)
        {
            for (int x = 0; x < src.cols; x++)
            {
                float dx = coeffs.evalDx(x, y);
                float dy = coeffs.evalDy(x, y);

                int x_src = (int)(x + dx + 0.5f);
                int y_src = (int)(y + dy + 0.5f);
                T val = 0;

                if ((x_src >= 0) && (x_src < src.cols) && (y_src >= 0) && (y_src < src.rows))
                {
                    val = src.at<T>(y_src, x_src);
                }

                dst.at<T>(y, x) = val;
            }
        }
    }

    /**
     * @brief Bi-linear sample a single location from an image. By GPT-4.
    */
    template <typename T>
    T bilinearSample(const cv::Mat& image, float x, float y) 
    {
        int x1 = static_cast<int>(x);
        int y1 = static_cast<int>(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        float dx = x - x1;
        float dy = y - y1;

        T I11 = image.at<T>(y1, x1);
        T I12 = image.at<T>(y2, x1);
        T I21 = image.at<T>(y1, x2);
        T I22 = image.at<T>(y2, x2);

        return static_cast<T>(
            (1 - dx) * (1 - dy) * I11 
            + (1 - dx) * dy * I12 
            + dx * (1 - dy) * I21 
            + dx * dy * I22);
    }

    /**
    * @brief Bi-linear sampling.
    */
    template <typename T>
    void remapPolyBilinear(const cv::Mat& src, RemapCoeffs3rdOrder& coeffs, cv::Mat& dst)
    {
        dst.create(src.rows, src.cols, src.type());

        for (int y = 0; y < src.rows; y++)
        {
            for (int x = 0; x < src.cols; x++)
            {
                T val = 0;

                float dx = coeffs.evalDx(x, y);
                float dy = coeffs.evalDy(x, y);

                float xf = x + dx;
                float yf = y + dy;

                if ((xf >= 0.0f) && (xf < src.cols - 1) && (yf >= 0.0f) && (yf < src.rows - 1))
                {
                    val = bilinearSample<T>(src, xf, yf);
                }

                dst.at<T>(y, x) = val;
            }
        }
    }

    float cubicInterpolate(float p0, float p1, float p2, float p3, float t);

    template <typename T>
    T bicubicSample(const cv::Mat& image, float x, float y)
    {
        int x1 = std::clamp(static_cast<int>(std::floor(x)) - 1, 0, image.cols - 1);
        int y1 = std::clamp(static_cast<int>(std::floor(y)) - 1, 0, image.rows - 1);
        int x2 = std::clamp(x1 + 1, 0, image.cols - 1);
        int y2 = std::clamp(y1 + 1, 0, image.rows - 1);
        int x3 = std::clamp(x1 + 2, 0, image.cols - 1);
        int y3 = std::clamp(y1 + 2, 0, image.rows - 1);
        int x4 = std::clamp(x1 + 3, 0, image.cols - 1);
        int y4 = std::clamp(y1 + 3, 0, image.rows - 1);

        float dx = x - x2;
        float dy = y - y2;

        std::vector<T> row_interpolations(4);

        for (int i = 0; i < 4; ++i)
        {
            float p1 = image.at<T>(y1 + i, x1);
            float p2 = image.at<T>(y1 + i, x2);
            float p3 = image.at<T>(y1 + i, x3);
            float p4 = image.at<T>(y1 + i, x4);

            row_interpolations[i] = static_cast<T>(cubicInterpolate(p1, p2, p3, p4, dx));
        }

        T interpolated_value = static_cast<T>(
            cubicInterpolate(row_interpolations[0], row_interpolations[1], row_interpolations[2], row_interpolations[3], dy));

        return interpolated_value;
    }

    /**
    * @brief Bicubic sampling.
    */
    template <typename T>
    void remapPolyBicubic(const cv::Mat& src, RemapCoeffs3rdOrder& coeffs, cv::Mat& dst)
    {
        dst.create(src.rows, src.cols, src.type());

        for (int y = 0; y < src.rows; y++)
        {
            for (int x = 0; x < src.cols; x++)
            {
                T val = 0;

                float dx = coeffs.evalDx(x, y);
                float dy = coeffs.evalDy(x, y);

                float xf = x + dx;
                float yf = y + dy;

                if ((xf >= 0.0f) && (xf < src.cols - 2) && (yf >= 0.0f) && (yf < src.rows - 2))
                {
                    val = bicubicSample<T>(src, xf, yf);
                }

                dst.at<T>(y, x) = val;
            }
        }
    }
}