//
// Created by shiyi on 2021/9/9.
//
#ifndef YOLOV5DET_H_
#define YOLOV5DET_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>

#include "NvInferRuntimeCommon.h"
#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include "common.h"
//#include "NvInferLegacyDims.h"

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif


class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) TRT_NOEXCEPT override
    {
        // suppress info-level messages
        if (severity == Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} ;
extern Logger gLogger;


class Dims5 : public nvinfer1::Dims
{
    public:
        //!
        //! \brief Construct an empty Dims5 object.
        //!
        Dims5()
                : nvinfer1::Dims{5, {}}
        {
        }

        //!
        //! \brief Construct a Dims5 from 5 elements.
        //!
        //! \param d0 The first element.
        //! \param d1 The second element.
        //! \param d2 The third element.
        //! \param d3 The fourth element.
        //! \param d4 The fifth element.
        //!
        Dims5(int32_t d0, int32_t d1, int32_t d2, int32_t d3, int32_t d4)
                : nvinfer1::Dims{5, {d0, d1, d2, d3, d4}}
        {
        }
} ;


struct YoloDetBox
{
    int x;
    int y;
    int width;
    int height;
    float score;
    int classId;
};


class YoloDet
{
    public:
        YoloDet(const char* planPath, int bs);
        virtual ~YoloDet();

        int doDet(const std::vector<cv::Mat*>& images,
                  std::vector<std::vector<YoloDetBox>>& inferOut);

    private:
        // Declaring static variables
        static int Channel;
        static int NewShape[2];
        static char* InputBlobName;
        static char* OutputBlobName;
        static char* OutputLAncBlobName;
        static char* OutputMAncBlobName;
        static char* OutputSAncBlobName;
        static int OutputClasses;
        static int OutputFeatures;
        static int AnchorNum;
        static int LargeFeatureH;
        static int LargeFeatureW;
        static int MediumFeatureH;
        static int MediumFeatureW;
        static int SmallFeatureH;
        static int SmallFeatureW;
        static float ConfThres;
        static float IouThres;
        static int Stride;

        int batchSize;
        int modelInSize;
        int modelOutSize;
        char* modelStream{nullptr};
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        int inputIndex;
        int outputIndex;
        int outputLargeFeaIndex;
        int outputMediumFeaIndex;
        int outputSmallFeaIndex;
        void* buffers[5];
        cudaStream_t stream;
        float* modelIn;
        float* modelOut;

        int preProcess(const std::vector<cv::Mat*>& images,
                       float* modelIn,
                       int batchSize, int channel, int newShape[2], int stride);

        int scaleFit(const cv::Mat& image,
                     cv::Mat& imagePadding,
                     cv::Scalar color, int newShape[2], int stride);

        int detect(float* modelIn,
                   float* modelOut,
                   int batchSize, int channel, int newShape[2], int outputFeatures, int outputClasses);

        int postProcess(float* modelOut,
                        std::vector<std::vector<Bbox>>& postOut,
                        const std::vector<cv::Mat*>& images, int batchSize, int newShape[2], int modelOutSize);
};

#endif //YOLOV5DET_H_
