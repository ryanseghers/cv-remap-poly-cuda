#include <string>
#include <chrono>
#include <ctime>
#include <future>

#include <fmt/core.h>
#include <opencv2/opencv.hpp>

#include "ImageUtil.h"
#include "remap.h"
#include "avxRemapPoly.h"

#include "cudaAddKernel.h"
#include "cudaUtil.h"
#include "cudaRemapPoly.h"

using namespace std;
using namespace std::chrono;
using namespace CVRemap;
using namespace CppOpenCVUtil;

std::array<float, 10> x_coeff =
{
    0.0, // c0
    0.0, // x
    0.0, // y
    0.0, // x^2
    0.0, // x * y
    0.0, // y^2
    0.0, // x^3
    0.0, // x^2 * y
    0.0,   // y^2 * x
    0.0, // y^3
};

std::array<float, 10> y_coeff =
{
    0.0, // c0
    0.0, // x
    0.0, // y
    0.0, // x^2
    0.0, // x * y
    0.0, // y^2
    0.0, // x^3
    0.0, // x^2 * y
    0.0,  // y^2 * x
    0.0, // y^3
};

void buildCvRemapImages(int width, int height, RemapCoeffs3rdOrder& coeffs, cv::Mat& dx, cv::Mat& dy)
{
    dx.create(height, width, CV_32F);
    dy.create(height, width, CV_32F);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            dx.at<float>(y, x) = x + coeffs.evalDx(x, y);
            dy.at<float>(y, x) = y + coeffs.evalDy(x, y);
        }
    }
}

void tryOpenCvRemap(cv::Mat& src, RemapCoeffs3rdOrder& coeffs, int samplingType, cv::Mat& dst)
{
    cv::Mat dx, dy;

    buildCvRemapImages(src.cols, src.rows, coeffs, dx, dy);
    saveDebugImage(dx, "dx-canonical");
    saveDebugImage(dy, "dy-canonical");

    buildRemapImagesAvx(src.cols, src.rows, coeffs, dx, dy);
    saveDebugImage(dx, "dx-avx");
    saveDebugImage(dy, "dy-avx");

    cv::remap(src, dst, dx, dy, samplingType);
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

    // build dx, dy images
    cv::Mat dx, dy;

    int passes = 0;
    float durationMs = 0.0;

    while (durationMs < 2000.0f)
    {
        if (doAvx)
        {
            buildRemapImagesAvx(src.cols, src.rows, coeffs, dx, dy);
        }
        else
        {
            buildCvRemapImages(src.cols, src.rows, coeffs, dx, dy);
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
    cv::Mat dx, dy;
    buildCvRemapImages(src.cols, src.rows, coeffs, dx, dy);

    // bench
    int passes = 0;
    float durationMs = 0.0;

    while (durationMs < 5000.0f)
    {
        cv::remap(src, dst, dx, dy, samplingType);
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

    //   coeffs.dxCoeffs = { 0.0f };
       //coeffs.dxCoeffs[0] = 3.0f;
       //coeffs.dyCoeffs = { 0.0f };
       //coeffs.dyCoeffs[0] = 4.0f;

    cv::Mat dst;

#ifdef _DEBUG
    remapPoly(img, coeffs, cv::INTER_NEAREST, dst);
    saveDebugImage(dst, "nearest");

    remapPoly16uAvx(img, coeffs, cv::INTER_NEAREST, dst);
    saveDebugImage(dst, "nearest_avx");

    return 0;

    remapPoly(img, coeffs, cv::INTER_LINEAR, dst);
    saveDebugImage(dst, "linear");

    remapPoly16uAvx(img, coeffs, cv::INTER_LINEAR, dst);
    saveDebugImage(dst, "linear_avx");

    //remapPoly(img, coeffs, cv::INTER_CUBIC, dst);
    //saveDebugImage(dst, "cubic");

    //tryOpenCvRemap(img, coeffs, cv::INTER_NEAREST, dst);
    //saveDebugImage(dst, "opencv_nearest");

    tryOpenCvRemap(img, coeffs, cv::INTER_LINEAR, dst);
    saveDebugImage(dst, "opencv_linear");

    //tryOpenCvRemap(img, coeffs, cv::INTER_CUBIC, dst);
    //saveDebugImage(dst, "opencv_cubic");

    setDevice(0);
    //printDeviceInfo();
    //tryCudaAddKernel();

    //tryCudaRemapPoly(img, coeffs, cv::INTER_NEAREST, dst);
    //saveDebugImage(dst, "cuda_nearest");

    tryCudaRemapPoly(img, coeffs, cv::INTER_LINEAR, dst);
    saveDebugImage(dst, "cuda_linear");

    //tryCudaRemapPoly(img, coeffs, cv::INTER_CUBIC, dst);
    //saveDebugImage(dst, "cuda_cubic");

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
