//
// Created by shiyi on 2021/10/22.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>

#include "NvInferRuntimeCommon.h"
#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include "NvInferPlugin.h"


#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif


class Logger : public nvinfer1::ILogger{

        void log(Severity severity, const char* msg) TRT_NOEXCEPT override
        {
            // suppress info-level messages
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
} gLogger;


const int BatchSize = 1;
const int NewShape[2] = {640, 384};
const int Stride = 32;
const char *planPath = "model.plan";
const char* InputBlobName = "images";
const char* OutputBlobName = "output";
const char* OutputBlobName1 = "outputFea1";
const char* OutputBlobName2 = "outputFea2";
const char* OutputBlobName3 = "outputFea3";
const int OutputClasses = 80;
const int OutputFeatures = 15120;


const float ConfThres = 0.25;
const float IouThres = 0.45;

int main() {

    int batchSize = BatchSize;
    int newShape[2] = {NewShape[0], NewShape[1]};
    int stride = Stride;
    int outputClasses = OutputClasses;
    int outputFeatures = OutputFeatures;
    std::ifstream planFile(planPath, std::ios::binary);
    size_t size = 0;

    planFile.seekg(0, planFile.end);
    size = planFile.tellg();
    planFile.seekg(0, planFile.beg);
    std::cout << "size: " << size << std::endl;
    char* modelStream = new char[size];
    assert(modelStream);
    planFile.read(modelStream, size);
    planFile.close();
    bool flag = initLibNvInferPlugins(&gLogger, "");
    std::cout << "flag: " << flag << std::endl;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream, size);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    std::cout << engine->getNbBindings() << std::endl;
    assert(engine->getNbBindings() == 5);
    int inputIndex = engine->getBindingIndex(InputBlobName);
    int outputIndex = engine->getBindingIndex(OutputBlobName);
    int outputIndex1 = engine->getBindingIndex(OutputBlobName1);
    int outputIndex2 = engine->getBindingIndex(OutputBlobName2);
    int outputIndex3 = engine->getBindingIndex(OutputBlobName3);

    float modelIn[batchSize * 3 * newShape[0] * newShape[1]];

//    float *modelOut = new float[batchSize * outputFea * (outputCls + 5)];
    float modelOut[batchSize * outputFeatures * ( outputClasses + 5 )];

    void* buffers[5];

    auto ret1 = cudaMalloc(&buffers[inputIndex], batchSize * 3 * newShape[0] * newShape[1] * sizeof(float));
    auto ret2 = cudaMalloc(&buffers[outputIndex], batchSize * outputFeatures * (outputClasses + 5) * sizeof(float));
    auto ret3 = cudaMalloc(&buffers[outputIndex1], batchSize * 3 * 48 * 80 * (outputClasses + 5) * sizeof(float));
    auto ret4 = cudaMalloc(&buffers[outputIndex2], batchSize * 3 * 24 * 40 * (outputClasses + 5) * sizeof(float));
    auto ret5 = cudaMalloc(&buffers[outputIndex3], batchSize * 3 * 12 * 20 * (outputClasses + 5) * sizeof(float));
    std::cout << "ret1: " << ret1 << std::endl;
    std::cout << "ret2: " << ret2 << std::endl;
    std::cout << "ret3: " << ret3 << std::endl;
    std::cout << "ret4: " << ret4 << std::endl;
    std::cout << "ret5: " << ret5 << std::endl;

    cudaStream_t stream;
    std::cout << "111111111111111111111111111" << std::endl;
    cudaMemcpyAsync(buffers[inputIndex],
                    modelIn,
                    batchSize * 3 * newShape[0] * newShape[1] * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream);
    std::cout << "2222222222222222222222222222222" << std::endl;
    bool status = context->enqueue(batchSize, buffers, stream, nullptr);
    std::cout << "status: " << status << std::endl;
    if (!status) {
        return 1;
    }
    cudaMemcpyAsync(modelOut,
                    buffers[outputIndex],
                    batchSize * outputFeatures * (outputClasses + 5),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    cudaFree(buffers[outputIndex1]);
    cudaFree(buffers[outputIndex2]);
    cudaFree(buffers[outputIndex3]);

    return 0;

}