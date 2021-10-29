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


extern const int BatchSize;
extern const int NewShape[2];
extern const int Stride;
extern const char* InputBlobName;
extern const char* OutputBlobName;
extern const int OutputClasses;
extern const int OutputFeatures;

extern const float ConfThres;
extern const float IouThres;

class Logger : public nvinfer1::ILogger{

        void log(Severity severity, const char* msg) TRT_NOEXCEPT override
        {
            // suppress info-level messages
            if (severity == Severity::kWARNING)
                std::cout << msg << std::endl;
        }
} ;
extern Logger gLogger;


struct DetBox {
    int x;
    int y;
    int width;
    int height;
    float score;
    int classId;
};


class Infer {

        char *modelStream{nullptr};
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        int inputIndex;
        int outputIndex;
        int outputIndex1;
        int outputIndex2;
        int outputIndex3;


    public:
        Infer(const char *planPath);
        virtual ~Infer();

        int doInfer(const std::vector<cv::Mat*> &images, std::vector<std::vector<DetBox>> &inferOut, int batchSize = BatchSize);

    private:
        int scaleFit(const cv::Mat &image, cv::Mat &imagePadding,
                     cv::Scalar color, int newShape[2], int stride);

        int preProcess(const std::vector<cv::Mat*> &images, float *modelIn, int newShape[2], int stride,
                       int batchSize = BatchSize);

        int detect(float* modelIn, float* modelOut, int newShape[2], int batchSize = BatchSize,
                   int outputFeatures = OutputFeatures, int outputClasses = OutputClasses);

        int postProcess(const std::vector<cv::Mat*> &images, float* modelOut, std::vector<std::vector<Bbox>> &postOut,
                        int newShape[2], int modelOutSize, int batchSize = BatchSize);
};

#endif //YOLOV5_INFER_H_
