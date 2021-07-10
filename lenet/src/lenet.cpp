//! \brief The lenet
// Created by shiyi on 2021/7/5.
//

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
        if (severity == Severity::kERROR)
            std::cout << msg << std::endl;
    }
} gLogger;

const int BATCH_SIZE = 1;
const int IMG_H = 28;
const int IMG_W = 28;
const int OUTPUT_SIZE = 10;
const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";


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
  const char onnx_filename[] = "../onnx/mnist_net.onnx";
  const char plan_filename[] = "../plan/lenet.plan";
  IBuilder* builder = createInferBuilder(gLogger);
  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
  parser->parseFromFile(onnx_filename, static_cast<int>(Logger::Severity::kWARNING));
  for (int i = 0; i < parser->getNbErrors(); ++i)
  {
    std::cout << parser->getError(i)->desc() << std::endl;
  }
  IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1 << 25);
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  IHostMemory* serializedModel = engine->serialize();
  std::ofstream p("../plan/lenet.plan", std::ios::binary);
  if (!p)
  {
    std::cerr << "could not open plan output file" << std::endl;
    return -1;
  }
  p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
  serializedModel->destroy();
  parser->destroy();
  IExecutionContext *context = engine->createExecutionContext();
  int input_index = engine->getBindingIndex(INPUT_BLOB_NAME);
  int output_index = engine->getBindingIndex(OUTPUT_BLOB_NAME);

  std::string test_img = "../data/MNIST/raw/t10k-images-idx3-ubyte";
  std::string test_label = "../data/MNIST/raw/t10k-labels-idx1-ubyte";
  static float input[BATCH_SIZE * 1 * IMG_H * IMG_W];
  static float output[BATCH_SIZE * OUTPUT_SIZE];
  int correct_num = 0;
  ReadMnist* readMnist = new ReadMnist;
//  ReadMnist readMnist;
  cv::Mat img = readMnist->readImg(test_img);
  cv::Mat labels = readMnist->readLabel(test_label);
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
  network->destroy();
  config->destroy();
  builder->destroy();

  return 0;
}