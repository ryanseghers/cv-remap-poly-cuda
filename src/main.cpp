#include <string>
#include <chrono>
#include <ctime>
#include <future>

#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "ImageUtil.h"
#include "remap.h"
#include "avxRemapPoly.h"

#include "cudaAddKernel.h"
#include "cudaUtil.h"
#include "cudaRemapPoly.h"
#include "brownConradyModel.h"

using namespace std;
using namespace std::chrono;
using namespace CVRemap;
using namespace CppOpenCVUtil;

std::array<float, 10> x_coeff = {-3.688481e+01, 8.223150e-02, 7.360715e-02, -6.642533e-05, -6.883254e-05, -3.595157e-05, 2.024929e-08, 2.200053e-09, 3.066782e-08, 8.405326e-10};
std::array<float, 10> y_coeff = { -3.688481e+01, 7.360715e-02, 8.223150e-02, -3.595157e-05, -6.883254e-05, -6.642533e-05, 8.405326e-10, 3.066782e-08, 2.200053e-09, 2.024929e-08};

void buildCvRemapImages(int width, int height, RemapCoeffs3rdOrder& coeffs, cv::Mat& xMap, cv::Mat& yMap)
{
    xMap.create(height, width, CV_32F);
    yMap.create(height, width, CV_32F);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            xMap.at<float>(y, x) = x + coeffs.evalDx(x, y);
            yMap.at<float>(y, x) = y + coeffs.evalDy(x, y);
        }
    }
}

void tryOpenCvRemap(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, int samplingType, cv::Mat& dst)
{
    cv::Mat xMap, yMap;

    buildCvRemapImages(src.cols, src.rows, coeffs, xMap, yMap);
    //saveDebugImage(xMap, "xMap-canonical");
    //saveDebugImage(yMap, "yMap-canonical");

    buildRemapImagesAvx(src.cols, src.rows, coeffs, xMap, yMap);
    //saveDebugImage(xMap, "xMap-avx");
    //saveDebugImage(yMap, "yMap-avx");

    cv::remap(src, dst, xMap, yMap, samplingType);
}

int benchRemapPoly(cv::Mat inSrc, RemapCoeffs3rdOrder coeffs, int samplingType, cv::Mat inDst, bool doAvx)
{
    auto tStart = steady_clock::now();

    // for benchmarking multi-threaded scenario, take copies of images
    cv::Mat src = inSrc.clone();
    cv::Mat dst = inDst.clone();

    int passes = 0;
    float durationMs = 0.0;

    while (durationMs < 5000.0f)
    {
        if (doAvx)
        {
            remapPoly16uAvx(src, coeffs, samplingType, dst);
        }
        else
        {
            remapPoly(src, coeffs, samplingType, dst);
        }

        passes++;

        auto tEnd = steady_clock::now() - tStart;
        durationMs = (float)duration_cast<milliseconds>(tEnd).count();
    }

    string sampleStr = samplingType == cv::INTER_NEAREST ? "nearest" : (samplingType == cv::INTER_LINEAR ? "linear" : "cubic");
    //fmt::print("CPU {}remap: Sampling: {}, iterations: {}, duration: {:.2f} ms\n", doAvx ? "AVX " : "", sampleStr, passes, durationMs / passes);

    return passes;
}

void benchBuildRemapImages(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, bool doAvx)
{
    auto tStart = steady_clock::now();

    // build x and y map images
    cv::Mat xMap, yMap;

    int passes = 0;
    float durationMs = 0.0;

    while (durationMs < 2000.0f)
    {
        if (doAvx)
        {
            buildRemapImagesAvx(src.cols, src.rows, coeffs, xMap, yMap);
        }
        else
        {
            buildCvRemapImages(src.cols, src.rows, coeffs, xMap, yMap);
        }

        passes++;

        auto tEnd = steady_clock::now() - tStart;
        durationMs = (float)duration_cast<milliseconds>(tEnd).count();
    }

    fmt::print("Build remap images{}: Duration: {:.2f} ms\n", doAvx ? " AVX" : "", durationMs / passes);
}

int benchOpenCvRemap(cv::Mat inSrc, RemapCoeffs3rdOrder coeffs, int samplingType, cv::Mat inDst)
{
    auto tStart = steady_clock::now();

    // for benchmarking multi-threaded scenario, take copies of images
    cv::Mat src = inSrc.clone();
    cv::Mat dst = inDst.clone();

    // build dx, dy images
    cv::Mat xMap, yMap;
    buildCvRemapImages(src.cols, src.rows, coeffs, xMap, yMap);

    // bench
    int passes = 0;
    float durationMs = 0.0;

    while (durationMs < 5000.0f)
    {
        cv::remap(src, dst, xMap, yMap, samplingType);
        passes++;

        auto tEnd = steady_clock::now() - tStart;
        durationMs = (float)duration_cast<milliseconds>(tEnd).count();
    }

    //string sampleStr = samplingType == cv::INTER_NEAREST ? "nearest" : (samplingType == cv::INTER_LINEAR ? "linear" : "cubic");
    //fmt::print("OpenCV remap: Sampling: {}, passes: {}, duration: {:.2f} ms\n", sampleStr, passes, durationMs / passes);

    return passes;
}

void tryCudaRemapPoly(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, int samplingType, cv::Mat& dst)
{
    dst.create(src.rows, src.cols, src.type());

    int width = src.cols;
    int height = src.rows;

    if (!src.isContinuous() || !dst.isContinuous())
    {
        throw std::runtime_error("The source and dest images must be continuous.");
    }

    uint16_t* psrc = src.ptr<uint16_t>();
    uint16_t* pdst = dst.ptr<uint16_t>();

    int deviceId = 0;
    cudaRemapPoly(deviceId, width, height, psrc, coeffs.dxCoeffs, coeffs.dyCoeffs, samplingType, pdst);
}

/**
 * @brief The sampling type barely matters. Cubic is a little bit slower (2.2 ms vs 2.0) but nearly same.
 * @param src
 * @param coeffs
 * @param samplingType
 * @param dst
*/
int benchCudaRemap(cv::Mat inSrc, RemapCoeffs3rdOrder coeffs, int samplingType, cv::Mat inDst)
{
    // for benchmarking multi-threaded scenario, take copies of images
    cv::Mat src = inSrc.clone();
    cv::Mat dst = inDst.clone();
    dst.create(src.rows, src.cols, src.type());

    auto tStart = steady_clock::now();
    int passes = 0;
    float durationMs = 0.0;

    // select device
    int deviceId = 0;
    setDevice(deviceId);

    int width = src.cols;
    int height = src.rows;

    if (!src.isContinuous() || !dst.isContinuous())
    {
        throw std::runtime_error("The source and dest images must be continuous.");
    }

    uint16_t* psrc = src.ptr<uint16_t>();
    uint16_t* pdst = dst.ptr<uint16_t>();

    while (durationMs < 100.0f)
    {
        cudaRemapPoly(deviceId, width, height, psrc, coeffs.dxCoeffs, coeffs.dyCoeffs, samplingType, pdst);
        passes++;

        auto tEnd = steady_clock::now() - tStart;
        durationMs = (float)duration_cast<milliseconds>(tEnd).count();
    }

    //string sampleStr = samplingType == cv::INTER_NEAREST ? "nearest" : (samplingType == cv::INTER_LINEAR ? "linear" : "cubic");
    //fmt::print("CUDA remap: Sampling: {}, Duration: {:.2f} ms\n", sampleStr, durationMs / passes);

    return passes;
}

/**
 * @brief Use std::async to run as many ops as possible, sweep number of simultaneous threads.
*/
void benchThroughput(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, cv::Mat& dst)
{
    // manually changing code to bench different functions
    string name = "cuda-nn";
    int samplingType = cv::INTER_NEAREST;
    bool doAvx = true;

    fmt::print("threads, {}\n", name);

    for (int nThreads = 1; nThreads <= 64; nThreads += (nThreads < 16 ? nThreads : 8))
    {
        // spawn a bunch of threads that each return how many passes they finished
        auto tStart = steady_clock::now();

        vector<future<int>> futures; // returns passes

        for (int i = 0; i < nThreads; i++)
        {
            //futures.push_back(std::async(benchRemapPoly, src, coeffs, samplingType, dst, doAvx));
            futures.push_back(std::async(benchOpenCvRemap, src, coeffs, samplingType, dst));
        }

        // wait and get results
        // (jagged ending could be an issue depending on how threads are scheduled...)
        int totalPasses = 0;

        for (int i = 0; i < futures.size(); i++)
        {
            totalPasses += futures[i].get();
        }

        auto tEnd = steady_clock::now() - tStart;
        float durationSeconds = (float)duration_cast<seconds>(tEnd).count();
        float throughput = totalPasses / durationSeconds;
        fmt::print("{}, {:.1f}\n", nThreads, throughput);
    }
}

/**
* @brief Do multi-parameter linear regression fitting of a surface using ordinary least squares.
* sample samplePoints A vector of sample points on the surface, where a sample point is a value corresponding to a location in an image.
* @param coeffs Output. The coefficients of a 2-d cubic polynomial:
*   f(x,y) = c0 + (c1 * x) + (c2 * y) + (c3 * x^2) + (c4 * xy) + (c5 * y^2) + (c6 * x^3) + (c7 * x^2*y) + (c8 * y^2*x) + (c9 * y^3)
* @return True for fitting succeeded, false for failed.
*/
bool fitSurface(const std::vector<std::tuple<cv::Point2f, float>>& samplePoints, std::array<float, 10>& coeffs)
{
    // Construct the design matrix
    int n = samplePoints.size();
    Eigen::MatrixXd X(n, 10);
    Eigen::VectorXd y(n);

    for (int i = 0; i < n; i++) {
        cv::Point2f pt = std::get<0>(samplePoints[i]);
        float z = std::get<1>(samplePoints[i]);
        X(i, 0) = 1;
        X(i, 1) = pt.x;
        X(i, 2) = pt.y;
        X(i, 3) = pt.x * pt.x;
        X(i, 4) = pt.x * pt.y;
        X(i, 5) = pt.y * pt.y;
        X(i, 6) = pt.x * pt.x * pt.x;
        X(i, 7) = pt.x * pt.x * pt.y;
        X(i, 8) = pt.x * pt.y * pt.y;
        X(i, 9) = pt.y * pt.y * pt.y;
        y(i) = z;
    }

    // Perform linear regression using OLS
    Eigen::VectorXd beta = X.colPivHouseholderQr().solve(y);

    // Copy the regression coefficients to the output array
    for (int i = 0; i < 10; i++) {
        coeffs[i] = beta(i);
    }

    return true;
}

void pointPairsToDeltaSurfaces(const std::vector<std::tuple<cv::Point2f, cv::Point2f>>& pointPairs, 
    std::vector<std::tuple<cv::Point2f, float>>& dxs, 
    std::vector<std::tuple<cv::Point2f, float>>& dys)
{
    for (const auto& pp: pointPairs)
    {
        cv::Point2f p1 = std::get<0>(pp);
        cv::Point2f p2 = std::get<1>(pp);
        dxs.push_back(make_tuple(p1, p2.x - p1.x));
        dys.push_back(make_tuple(p1, p2.y - p1.y));
    }
}

/**
 * @brief Do multi-parameter linear regression fitting using ordinary least squares on the 
 * @param pointPairs 
 * @param coeffs 
 * @return 
*/
bool fitPointPairs(const std::vector<std::tuple<cv::Point2f, cv::Point2f>>& pointPairs, RemapCoeffs3rdOrder& coeffs)
{
    vector<tuple<cv::Point2f, float>> dxs, dys;
    pointPairsToDeltaSurfaces(pointPairs, dxs, dys);
    return fitSurface(dxs, coeffs.dxCoeffs) && fitSurface(dys, coeffs.dyCoeffs);
}

/**
 * @brief Convert a 16-bit grayscale image to an 8-bit RGB image by mapping the 1st and 99th percentiles to 0 and 255, respectively.
 * @param img16u 
 * @param dstRgb 
 * @param lowPercentile 
 * @param highPercentile 
*/
void convert16uToRgbByPercentiles(cv::Mat& img16u, cv::Mat& dstRgb, float lowPercentile=1.0f, float highPercentile=99.0f)
{
    double min, max;
    cv::minMaxLoc(img16u, &min, &max);
    float low = min + (max - min) * lowPercentile / 100.0f;
    float high = min + (max - min) * highPercentile / 100.0f;
    cv::Mat img8u;
    img16u.convertTo(img8u, CV_8U, 255.0f / (high - low), -255.0f * low / (high - low));
    cv::cvtColor(img8u, dstRgb, cv::COLOR_GRAY2RGB);
}

void renderArrows(cv::Mat& img, const std::vector<std::tuple<cv::Point2f, cv::Point2f>>& pointPairs)
{
    for (const auto& pp : pointPairs)
    {
        cv::Point2f p1 = std::get<0>(pp);
        cv::Point2f p2 = std::get<1>(pp);
        cv::arrowedLine(img, p1, p2, cv::Scalar(0, 255, 255), 1);
    }
}

void tryBrownConradyModel(cv::Mat& src)
{
    int width = src.cols;
    int height = src.rows;

    BrownConradyCoeffs coeffs;
    coeffs.radial = { -0.025f, 0.01f, 0.01f, 0.0f, 0.0f, 0.0f };

    cv::Mat xMap, yMap;
    buildBrownConradyRemapImages(width, height, coeffs, xMap, yMap);
    //saveDebugImage(xMap, "brown-conrady-x");
    //saveDebugImage(yMap, "brown-conrady-y");

    cv::Mat dst;
    applyBrownConradyModel(src, coeffs, cv::INTER_LINEAR, dst);
    saveDebugImage(dst, "brown-conrady-out");

    // try fitting the brown-conrady model with the cubic poly
    // eval brown-conrady at grid of sample points
    const int gridRes = 128;
    vector<tuple<cv::Point2f, cv::Point2f>> pointPairs;

    for (int y = 0; y < height; y += gridRes)
    {
        for (int x = 0; x < width; x += gridRes)
        {
            cv::Point2i srcPt = { x, y };
            cv::Point2f distPt = applyBrownConradyModel(width, height, x, y, coeffs);
            pointPairs.push_back(std::make_tuple(srcPt, distPt));
        }
    }

    cv::Mat tmp;
    convert16uToRgbByPercentiles(src, tmp);
    renderArrows(tmp, pointPairs);
    saveDebugImage(tmp, "point-pairs");

    RemapCoeffs3rdOrder coeffs3;
    
    if (fitPointPairs(pointPairs, coeffs3))
    {
        printf("Fitting point pairs succeeded.\n");
        coeffs3.print();

        remapPolyBilinear<uint16_t>(src, coeffs3, dst);
        saveDebugImage(dst, "poly-fit-out");
    }
    else
    {
        printf("Fitting point pairs failed.\n");
    }
}

int main()
{
    fmt::print("Starting...\n");
    ImageUtil::init();

    // load test image
    string imgPath = "C:/Temp/vm-share/wxiv-test-images/FC10880_LANEA_CYCLE000_STEP1.snap_0_STAGE000_488.tif";
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);

    saveDebugImage(img, "orig");

    // remap
    RemapCoeffs3rdOrder coeffs;
    coeffs.dxCoeffs = x_coeff;
    coeffs.dyCoeffs = y_coeff;

    cv::Mat dst;

    //tryBrownConradyModel(img);

#ifdef _DEBUG
    //remapPoly(img, coeffs, cv::INTER_NEAREST, dst);
    //saveDebugImage(dst, "nearest");

    //remapPoly16uAvx(img, coeffs, cv::INTER_NEAREST, dst);
    //saveDebugImage(dst, "nearest_avx");

    //remapPoly(img, coeffs, cv::INTER_LINEAR, dst);
    //saveDebugImage(dst, "linear");

    //remapPoly(img, coeffs, cv::INTER_CUBIC, dst);
    //saveDebugImage(dst, "cubic");

    //tryOpenCvRemap(img, coeffs, cv::INTER_NEAREST, dst);
    //saveDebugImage(dst, "opencv_nearest");

    //tryOpenCvRemap(img, coeffs, cv::INTER_LINEAR, dst);
    //saveDebugImage(dst, "opencv_linear");

    tryOpenCvRemap(img, coeffs, cv::INTER_CUBIC, dst);
    saveDebugImage(dst, "opencv_cubic");

    setDevice(0);
    //printDeviceInfo();
    //tryCudaAddKernel();

    //tryCudaRemapPoly(img, coeffs, cv::INTER_NEAREST, dst);
    //saveDebugImage(dst, "cuda_nearest");

    //tryCudaRemapPoly(img, coeffs, cv::INTER_LINEAR, dst);
    //saveDebugImage(dst, "cuda_linear");

    tryCudaRemapPoly(img, coeffs, cv::INTER_CUBIC, dst);
    saveDebugImage(dst, "cuda_cubic");

#else
    //benchBuildRemapImages(img, coeffs, false);
    //benchBuildRemapImages(img, coeffs, true);

    //benchRemapPoly(img, coeffs, cv::INTER_NEAREST, dst, false);
    //benchRemapPoly(img, coeffs, cv::INTER_NEAREST, dst, true);

    benchThroughput(img, coeffs, dst);

    // bench
    //vector<int> sampleTypes = { cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC };

    //for (int sampleType : sampleTypes)
    //{
    //    benchRemapPoly(img, coeffs, sampleType, dst);
    //}

    //for (int sampleType : sampleTypes)
    //{
    //    benchOpenCvRemap(img, coeffs, sampleType, dst);
    //}

    //for (int sampleType : sampleTypes)
    //{
    //    benchCudaRemap(img, coeffs, sampleType, dst);
    //}

    //benchCudaRemap(img, coeffs, cv::INTER_CUBIC, dst);

#endif

    return 0;
}
