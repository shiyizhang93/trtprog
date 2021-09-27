//
// Created by shiyi on 2021/9/9.
//

#include "infer.h"


Infer::Infer(const char *planPath) {
    std::ifstream planFile(planPath, std::ios::binary);
    size_t size{0};
    planFile.seekg(0, planFile.end);
    size = planFile.tellg();
    planFile.seekg(0, planFile.beg);
    modelStream = new char[size];
    planFile.read(modelStream, size);
    planFile.close();
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream, size, nullptr);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    inputIndex = engine->getBindingIndex(InputBlobName);
    outputIndex = engine->getBindingIndex(OutputBlobName);
}


Infer::~Infer() {
    context->destroy();
    engine->destroy();
    delete []modelStream;
}


int Infer::doInfer(const std::vector<cv::Mat> &images, std::vector<std::vector<DetBox>> *box) {
    // Get the local variable value of NewShape and Stride
    int newShape[2] = {NewShape[0], NewShape[1]};
    int stride = Stride;
    int batchSize = images.size();
    float modelIn[batchSize * 3 * newShape[0] * newShape[1] * sizeof(float)];
    float modelOut[batchSize * OutputFeatures * (OutputClasses + 5) * sizeof(float)];
    // Do pre-process
    preProcess(images, modelIn, newShape, stride, batchSize);
    // Pass input data to model and get output data
    detect(images, modelIn, modelOut, newShape, batchSize);
    // Do post-process

}


int Infer::scaleFit(const cv::Mat &image, cv::Mat &imagePadding, cv::Scalar color, int newShape[2], int stride) {
    // Get the original shape of image
    int shape[2] = {image.size().height, image.size().width}; // current shape {height, width}
    // Get the minimum fraction ratio
    float ratio = fmin(newShape[0] / shape[0], newShape[1] / shape[1]);
    // Compute padding {width, height}
    int newUnpad[2] = {(int) round(newShape[1] * ratio), (int) round(newShape[0] * ratio)};
    int dw = ((newShape[1] - newUnpad[0]) % Stride) / 2;
    int dh = (newShape[0] - newUnpad[1]) % Stride / 2;
    cv::Mat img;
    if (shape[0] != newShape[0] && shape[1] != newShape[1]){
        cv::resize(image, img, cv::Size(newUnpad[0], newUnpad[1]));
    }
    int top = (int) round(dh - 0.1);
    int bottom = (int) round(dh + 0.1);
    int left = (int) round(dw - 0.1);
    int right = (int) round(dw + 0.1);
    cv::copyMakeBorder(img, imagePadding, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return 0;
}


int Infer::preProcess(const std::vector<cv::Mat> &images, float *modelIn, int newShape[2], int stride, int batchSize) {
    cv::Scalar color = cv::Scalar(114, 114, 114);
    for (int b = 0; b < batchSize; b++){
        cv::Mat imagePadding;
        scaleFit(images[b], imagePadding, color, newShape, stride);
        int i = 0;
        for (int r = 0; r < imagePadding.rows; r++){
            uchar* ucPixel = imagePadding.data + r * imagePadding.step;
            for (int c = 0; c < imagePadding.cols; c++){
                modelIn[batchSize * 3 * imagePadding.rows * imagePadding.cols + i] = (float)ucPixel[2] / 255.0;
                modelIn[batchSize * 3 * imagePadding.rows * imagePadding.cols + i + imagePadding.rows * imagePadding.cols] = (float)ucPixel[1] / 255.0;
                modelIn[batchSize * 3 * imagePadding.rows * imagePadding.cols + i + 2 * imagePadding.rows * imagePadding.cols] = (float)ucPixel[0] / 255.0;
                ucPixel += 3;
                i++;
            }
        }
    }
    return 0;
}


int Infer::detect(const std::vector<cv::Mat> &images, float* modelIn, float* modelOut, int newShape[2], int batchSize) {

    // Run inference
    auto start = std::chrono::system_clock::now();
    void * buffers[2];
    cudaMalloc(&buffers[inputIndex], batchSize * 3 * newShape[0] * newShape[1] * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * OutputFeatures * (OutputClasses + 5) * sizeof(float));
    cudaStream_t stream;
    cudaMemcpyAsync(buffers[inputIndex], modelIn,
                    batchSize * 3 * newShape[0] * newShape[1] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    context->enqueue(batchSize, buffers, stream, nullptr);
    cudaMemcpyAsync(modelOut, buffers[outputIndex],
                    batchSize * OutputFeatures * (OutputClasses + 5),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    auto end = std::chrono::system_clock::now();

    return 0;
}
