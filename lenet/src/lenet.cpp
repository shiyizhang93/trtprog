//! \Brief  This file contains the implementation of the ONNX MNIST sample. I creates the network using the
//! MNIST onnx modle
//!
//! \Author Shiyi Zhang
//!
//! \Create 07/05/2021
//!
//! \Modify 07/12/2021


#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "ReadMnist.h"

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) TRT_NOEXCEPT override
    {
        // suppress info-level messages
        if (severity == Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

const int BATCH_SIZE = 1;
const int IMG_H = 28;
const int IMG_W = 28;
const int OUTPUT_SIZE = 10;
const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";

const char onnx_path[] = "../onnx/mnist_net.onnx";
const char plan_path[] = "../plan/lenet.plan";
std::string test_img = "../data/MNIST/raw/t10k-images-idx3-ubyte";
std::string test_label = "../data/MNIST/raw/t10k-labels-idx1-ubyte";
static float input[BATCH_SIZE * 1 * IMG_H * IMG_W];
static float output[BATCH_SIZE * OUTPUT_SIZE];


int buildEngine()
{
    IBuilder* builder = createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(onnx_path, static_cast<int>(Logger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 25);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    IHostMemory* serialized_model = engine->serialize();
    std::ofstream plan_file(plan_path, std::ios::binary);
    if (!plan_file)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    plan_file.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());

    serialized_model->destroy();
    engine->destroy();
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

    return 0;
}


ICudaEngine* deserializeEngine()
{
    std::ifstream plan_file(plan_path, std::ios::binary);
    char *model_stream{nullptr};
    size_t size{0};

    plan_file.seekg(0, plan_file.end);
    size = plan_file.tellg();
    plan_file.seekg(0, plan_file.beg);
    model_stream = new char[size];
    assert(model_stream);
    plan_file.read(model_stream, size);
    plan_file.close();
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(model_stream, size, nullptr);
    return engine;
}


void doInference(IExecutionContext& context, float* input, float* output, int input_index, int output_index)
{
    void* buffers[2];
    cudaMalloc(&buffers[input_index],  BATCH_SIZE * 1 * IMG_H * IMG_W * sizeof(float));
    cudaMalloc(&buffers[output_index], BATCH_SIZE * 1 * OUTPUT_SIZE * sizeof(float));
    cudaStream_t stream;
    cudaMemcpyAsync(buffers[input_index], input,
                  BATCH_SIZE * 1 * IMG_H * IMG_W * sizeof(float), cudaMemcpyHostToDevice, stream);
    context.enqueue(BATCH_SIZE, buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[output_index],
                  BATCH_SIZE * 1 * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output_index]);
}


void softmax(float *x)
{
    float max = 0.0;
    float sum = 0.0;
    // Get the hard max value
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        if (max < x[i])
            max = x[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        x[i] = std::exp(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        x[i] /= sum;
    }
}


int main(int argc, char** argv)
{
    std::ifstream plan_file(plan_path, std::ios::binary);
    if(!plan_file.good())
        buildEngine();
    ICudaEngine* engine = deserializeEngine();
    IExecutionContext *context = engine->createExecutionContext();
    int input_index = engine->getBindingIndex(INPUT_BLOB_NAME);
    int output_index = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    // New an instance for processing MNIST dataset
    ReadMnist* readMnist = new ReadMnist;
    cv::Mat img = readMnist->readImg(test_img);
    cv::Mat labels = readMnist->readLabel(test_label);
    int correct_num = 0;
    for (int i = 0; i < img.rows; i++)
    {
        uchar* ptr = img.ptr<uchar>(i);
        uchar* ptr_label = labels.ptr<uchar>(i);
        for (int j = 0; j < img.cols * img.channels(); j++)
        {
            input[j] = (((float)ptr[j] / 255.0) - 0.5) / 0.5;
        }
        doInference(*context, input, output, input_index, output_index);
        softmax(output);
        int maxPos = std::max_element(output, output+OUTPUT_SIZE) - output;
        int label = (int)ptr_label[0];
        if (maxPos == label)
            correct_num += 1;
    }
    std::cout << (float)correct_num / img.rows << std::endl;

    delete [] readMnist;
    context->destroy();
    engine->destroy();

  return 0;
}
