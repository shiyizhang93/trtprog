//
// Created by shiyi on 2021/9/9.
//

#include "yolov5Det.h"


Logger gLogger;

// Initialize static class member variables
int YoloDet::Channel = 3;
int YoloDet::NewShape[2] = {384, 640};
int YoloDet::OutputClasses = 80;
int YoloDet::OutputFeatures = 15120;
int YoloDet::AnchorNum = 3;
int YoloDet::LargeFeatureH = 48;
int YoloDet::LargeFeatureW = 80;
int YoloDet::MediumFeatureH = 24;
int YoloDet::MediumFeatureW = 40;
int YoloDet::SmallFeatureH = 12;
int YoloDet::SmallFeatureW = 20;
float YoloDet::ConfThres = 0.25;
float YoloDet::IouThres = 0.45;
char *YoloDet::InputBlobName = "images";
char *YoloDet::OutputBlobName = "output";
char *YoloDet::OutputLAncBlobName = "output_l";
char *YoloDet::OutputMAncBlobName = "output_m";
char *YoloDet::OutputSAncBlobName = "output_s";
int YoloDet::Stride = 32;


YoloDet::YoloDet(const char* planPath, int bs)
{
    // Assign the value of static member to local variable
    int channel = Channel;
    auto newShape = NewShape;
    int outputClasses = OutputClasses;
    int outputFeatures = OutputFeatures;
    int anchorNum = AnchorNum;
    int largeFeatureH = LargeFeatureH;
    int largeFeatureW = LargeFeatureW;
    int mediumFeatureH = MediumFeatureH;
    int mediumFeatureW = MediumFeatureW;
    int smallFeatureH = SmallFeatureH;
    int smallFeatureW = SmallFeatureW;
    // Assign batch size user input as class private member variable
    batchSize = bs;
    // Initialize class private member variable modelInSize, modelOutSize, modelIn and modelOut
    modelInSize = batchSize * channel * newShape[0] * newShape[1];
    modelOutSize = batchSize * outputFeatures * (outputClasses + 5);
    modelIn = new float[modelInSize];
    modelOut = new float[modelOutSize];
    // Initialize local variables
    size_t size = 0;

    // Read TensorRT plan file
    std::ifstream planFile(planPath, std::ios::binary);
    planFile.seekg(0, planFile.end);
    size = planFile.tellg();
    planFile.seekg(0, planFile.beg);
    // Initialize a stream to store plan file info
    modelStream = new char[size];
    assert(modelStream);
    planFile.read(modelStream, size);
    planFile.close();

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(modelStream, size);
    context = engine->createExecutionContext();
    std::cout << engine->getNbBindings() << std::endl;
    assert(engine->getNbBindings() == 5);
    inputIndex = engine->getBindingIndex(InputBlobName);
    outputIndex = engine->getBindingIndex(OutputBlobName);
    outputLargeFeaIndex = engine->getBindingIndex(OutputLAncBlobName);
    outputMediumFeaIndex = engine->getBindingIndex(OutputMAncBlobName);
    outputSmallFeaIndex = engine->getBindingIndex(OutputSAncBlobName);
    context->setBindingDimensions(inputIndex,
                                  nvinfer1::Dims4(batchSize, channel, newShape[0], newShape[1]));
    context->setBindingDimensions(outputIndex,
                                  nvinfer1::Dims3(batchSize, outputFeatures, outputClasses));
    context->setBindingDimensions(outputLargeFeaIndex,
                                  nvinfer1::Dims4(batchSize, anchorNum, largeFeatureH * largeFeatureW, outputClasses));
    context->setBindingDimensions(outputMediumFeaIndex,
                                  nvinfer1::Dims4(batchSize, anchorNum, mediumFeatureH * mediumFeatureW, outputClasses));
    context->setBindingDimensions(outputSmallFeaIndex,
                                  nvinfer1::Dims4(batchSize, anchorNum, smallFeatureH * smallFeatureW, outputClasses));
    assert(engine->getNbBindings() == 5);
    cudaMalloc(&buffers[inputIndex], batchSize * channel * newShape[0] * newShape[1]);
    cudaMalloc(&buffers[outputIndex], batchSize * outputFeatures * outputClasses);
    cudaMalloc(&buffers[outputLargeFeaIndex], batchSize * anchorNum * largeFeatureH * largeFeatureW * outputClasses);
    cudaMalloc(&buffers[outputMediumFeaIndex], batchSize * anchorNum * mediumFeatureH * mediumFeatureW * outputClasses);
    cudaMalloc(&buffers[outputSmallFeaIndex], batchSize * anchorNum * smallFeatureH * smallFeatureW * outputClasses);

    cudaStreamCreate(&stream);
    std::cout << "The number of optimization profiles" << engine->getNbOptimizationProfiles() << std::endl;
    context->setOptimizationProfileAsync(0, stream);
}


YoloDet::~YoloDet()
{
    delete[] modelStream;
    modelStream = nullptr;
    delete[] modelIn;
    modelIn = nullptr;
    delete[] modelOut;
    modelOut = nullptr;
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    cudaFree(buffers[outputLargeFeaIndex]);
    cudaFree(buffers[outputMediumFeaIndex]);
    cudaFree(buffers[outputSmallFeaIndex]);
}


int YoloDet::doDet(const std::vector<cv::Mat*>& images,
                   std::vector<std::vector<YoloDetBox>>& inferOut)
{
    // Get the local variable value of NewShape and Stride
    int bs = batchSize;
    int channel = Channel;
    auto newShape = NewShape;
    int outputFea = OutputFeatures;
    int outputCls = OutputClasses;
    int stride = Stride;

    std::vector<std::vector<Bbox>> postOut;
    YoloDetBox detBox;
    std::vector<YoloDetBox> boxes;
    // Do pre-process
    preProcess(images, modelIn, bs, channel, newShape, stride);
    // Pass input data to model and get output data
    detect(modelIn, modelOut, bs, channel, newShape, outputFea, outputCls);
    // Do post-process
    postProcess(modelOut, postOut, images, bs, newShape, modelOutSize);

   for (int b = 0; b < batchSize; b++)
   {
       for (int i = 0; i < postOut[b].size(); i++)
       {
           detBox.x = postOut[b][i].rect.x;
           detBox.y = postOut[b][i].rect.y;
           detBox.width = postOut[b][i].rect.width;
           detBox.height = postOut[b][i].rect.height;
           detBox.score = postOut[b][i].conf;
           detBox.classId = postOut[b][i].cls;
           boxes.push_back(detBox);
       }
       inferOut.push_back(boxes);
    }

   return 0;
}


int YoloDet::preProcess(const std::vector<cv::Mat*>& images,
                        float* modelIn,
                        int batchSize, int channel, int newShape[2], int stride)
{
    cv::Scalar color = cv::Scalar(114, 114, 114);
    for (int b = 0; b < batchSize; b++)
    {
        cv::Mat imagePadding;
        scaleFit(*images[b], imagePadding, color, newShape, stride);
        int i = 0;
        for (int r = 0; r < imagePadding.rows; r++)
        {
            uchar* ucPixel = imagePadding.data + r * imagePadding.step;
            for (int c = 0; c < imagePadding.cols; c++)
            {
                modelIn[b * channel * imagePadding.rows * imagePadding.cols + i] = (float)ucPixel[2] / 255.0;
                modelIn[b * channel * imagePadding.rows * imagePadding.cols + i + imagePadding.rows * imagePadding.cols] = (float)ucPixel[1] / 255.0;
                modelIn[b * channel * imagePadding.rows * imagePadding.cols + i + 2 * imagePadding.rows * imagePadding.cols] = (float)ucPixel[0] / 255.0;
                ucPixel += channel;
                i++;
            }
        }
    }

    return 0;
}


int YoloDet::scaleFit(const cv::Mat& image,
                      cv::Mat& imagePadding,
                      cv::Scalar color, int newShape[2], int stride)
{
    // Get the original shape of image
    int shape[2] = {image.size().height, image.size().width}; // current shape {height, width}
    // Get the minimum fraction ratio
    float ratio = fmin((float) newShape[0] / shape[0], (float) newShape[1] / shape[1]);
    std::cout << newShape[0] << "," << newShape[1] << std::endl;
    // Compute padding {height, width}
    int newUnpad[2] = {(int) round(shape[0] * ratio), (int) round(shape[1] * ratio)};
    std::cout << newUnpad[0] << "," << newUnpad[1] << std::endl;
    int dh = ((newShape[0] - newUnpad[0]) % stride) / 2;
    int dw = ((newShape[1] - newUnpad[1]) % stride) / 2;

    cv::Mat img;
    if (shape[0] != newShape[0] && shape[1] != newShape[1])
    {
        cv::resize(image, img, cv::Size(newUnpad[1], newUnpad[0]));
    }
    int top = (int) round(dh - 0.1);
    int bottom = (int) round(dh + 0.1);
    int left = (int) round(dw - 0.1);
    int right = (int) round(dw + 0.1);
    cv::copyMakeBorder(img, imagePadding, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return 0;
}


int YoloDet::detect(float* modelIn,
                    float* modelOut,
                    int batchSize, int channel, int newShape[2], int outputFeatures, int outputClasses)
{
    // Run inference
    cudaMemcpyAsync(buffers[inputIndex],
                    modelIn,
                    batchSize * channel * newShape[0] * newShape[1] * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream);
    bool status = context->enqueue(batchSize, buffers, stream, nullptr);
    std::cout << "status: " << status << std::endl;
    if (!status)
    {
        return 1;
    }
    cudaMemcpyAsync(modelOut,
                    buffers[outputIndex],
                    batchSize * outputFeatures * (outputClasses + 5) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return 0;
}


int YoloDet::postProcess(float* modelOut,
                         std::vector<std::vector<Bbox>> &postOut,
                         const std::vector<cv::Mat*>& images, int batchSize, int newShape[2], int modelOutSize)
{
    std::vector<Bbox> nmsBboxes;
    std::vector<Bbox> scaleBboxes;
    for (int b = 0; b < batchSize; b++)
    {
        // Init intermediate variables
        int img0Shape[2] = {images[b]->rows, images[b]->cols};
        nonMaxSuppression(modelOut, nmsBboxes, ConfThres, IouThres, b, OutputClasses, modelOutSize);
        scaleCoords(nmsBboxes, scaleBboxes, newShape, img0Shape);
        postOut.push_back(scaleBboxes);
        nmsBboxes.clear();
        scaleBboxes.clear();
    }

    return 0;
}