#include "brownConradyModel.h"

namespace CVRemap
{
    /**
    * @brief Apply the specified Brown-Conrady coefficients to the specified input pixel location to produce
    * the distorted or undistorted location.
    * Mostly by GPT-4.
    */
    cv::Point2f applyBrownConradyModel(int width, int height, int xi, int yi, const BrownConradyCoeffs& coeffs)
    {
        // normalize image coordinates to range [-1,1]
        int cx = width / 2;
        int cy = height / 2;
        float x = (float)(xi - cx) / cx;
        float y = (float)(yi - cy) / cy;
        float r2 = x * x + y * y;

        float radialDistortion = 1.0f;

        if ((xi != cx) || (yi != cy))
        {
            // radial distortion
            float r4 = r2 * r2;
            float r6 = r2 * r4;
            radialDistortion = 1 + coeffs.radial[0] * r2 + coeffs.radial[1] * r4 + coeffs.radial[2] * r6;
        }

        // tangential distortion
        float tangentialDistortion_x = 2*coeffs.p1*x*y + coeffs.p2*(r2 + 2*x*x);
        float tangentialDistortion_y = coeffs.p1*(r2 + 2*y*y) + 2*coeffs.p2*x*y;

        // apply distortion to x and y coordinates
        float xd = x*radialDistortion + tangentialDistortion_x;
        float yd = y*radialDistortion + tangentialDistortion_y;

        // back to pixel coordinates
        xd = xd * cx + cx;
        yd = yd * cy + cy;

        return cv::Point2f(xd, yd);
    }

    void buildBrownConradyRemapImages(int width, int height, const BrownConradyCoeffs& coeffs, cv::Mat& xMap, cv::Mat& yMap)
    {
        xMap.create(height, width, CV_32F);
        yMap.create(height, width, CV_32F);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                cv::Point2f pt = applyBrownConradyModel(width, height, x, y, coeffs);
                xMap.at<float>(y, x) = pt.x;
                yMap.at<float>(y, x) = pt.y;
            }
        }
    }

    /**
    * @brief Apply the specified Brown-Conrady coefficients to the source image to distort or undistort it.
    * This uses OpenCV remap.
    * @param samplingType OpenCV constant for interpolation type, e.g. cv::INTER_NEAREST, cv::INTER_LINEAR,  
    */
    void applyBrownConradyModel(cv::Mat& src, const BrownConradyCoeffs& coeffs, int samplingType, cv::Mat& dst)
    {
        cv::Mat xMap, yMap;
        buildBrownConradyRemapImages(src.cols, src.rows, coeffs, xMap, yMap);
        cv::remap(src, dst, xMap, yMap, samplingType);
    }
}
