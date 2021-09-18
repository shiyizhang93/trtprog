//
// Created by shiyi on 2021/9/9.
//

#ifndef YOLOV5_YOLOV5_H_
#define YOLOV5_YOLOV5_H_

#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>

#include "NvInferRuntimeCommon.h"
#include "opencv2/opencv.hpp"
#include "NvInfer.h"


const int NewShape[2] = {640, 640};
const int Stride = 32;
const char* InputBlobName = "images";
const char* OutputBlobName = "Output";
const int OutputClasses = 80;
const int OutputFeatures = 25200;


struct DetBox {
    int x;
    int y;
    int width;
    int height;
    float score;
    int class_id;
};


class Logger : public ILogger{

        void log(Severity severity, const char* msg) TRT_NOEXCEPT override
                {
                        // suppress info-level messages
                        if (severity == Severity::kWARNING)
                        std::cout << msg << std::endl;
                }
} gLogger;


class Infer {

        char *modelStream{nullptr};
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        int inputIndex;
        int outputIndex;

    public:
        Infer(const std::string &planPath);
        virtual ~Infer();

        int doInfer(const std::vector<cv::Mat> &images, std::vector<std::vector<DetBox>> *box);

    private:
        int scaleFit(cv::Mat &image, cv::Mat &imagePadding, cv::Scalar color;
        int preProcess(const std::vector<cv::Mat> images, float *modelIn);
        int detect(const std::vector<cv::Mat> &images, float* modelIn, float* modelOut);
        int postProcess(float* modelOut);
        void nms();
        float iou();
        bool cmp();
};

#endif //YOLOV5_YOLOV5_H_
