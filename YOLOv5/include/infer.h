//
// Created by shiyi on 2021/9/9.
//
#ifndef YOLOV5_INFER_H_
#define YOLOV5_INFER_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>

#include "NvInferRuntimeCommon.h"
#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include "common.h"

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif


//extern const int BatchSize;
//extern const int NewShape[2];
//extern const int Stride;
//extern const char* InputBlobName;
//extern const char* OutputBlobName;
//extern const int OutputClasses;
//extern const int OutputFeatures;
//
//extern const float ConfThres;
//extern const float IouThres;

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
        YoloDet(const char *planPath, int bs);
        virtual ~YoloDet();

        int doDet(const std::vector<cv::Mat*> &images,
                  std::vector<std::vector<YoloDetBox>> &inferOut);

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
        static float ConfThres;
        static float IouThres;
        static int Stride;

        int batchSize;
        char* modelStream{nullptr};
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        int inputIndex;
        int outputIndex;
        int outputLAncIndex;
        int outputMAncIndex;
        int outputSAncIndex;
        void* buffers[5];
        cudaStream_t stream;
        float* modelIn;
        float* modelOut;

        int preProcess(const std::vector<cv::Mat*>& images,
                       float* modelIn,
                       int channel, int newShape[2], int stride);

        int scaleFit(const cv::Mat& image,
                     cv::Mat& imagePadding,
                     cv::Scalar color, int newShape[2], int stride);

        int detect(float* modelIn,
                   float* modelOut,
                   int channel, int newShape[2], int outputFeatures = OutputFeatures, int outputClasses = OutputClasses);

        int postProcess(float* modelOut,
                        std::vector<std::vector<Bbox>> &postOut,
                        const std::vector<cv::Mat*>& images, int channel, int newShape[2], int modelOutSize);
};

#endif //YOLOV5_INFER_H_
