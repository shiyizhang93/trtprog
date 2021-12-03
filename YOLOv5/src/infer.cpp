//
// Created by shiyi on 2021/9/9.
//

#include "infer.h"
#include "NvInferPlugin.h"


Logger gLogger;

// Initialize static class member variables
int YoloDet::Channel = 3;
int YoloDet::NewShape[2] = {384, 640};
int YoloDet::OutputClasses = 80;
int YoloDet::OutputFeatures = 25200;
float YoloDet::ConfThres = 0.25;
float YoloDet::IouThres = 0.45;
char *YoloDet::InputBlobName = "images";
char *YoloDet::OutputBlobName = "output";
char *YoloDet::OutputLAncBlobName = "output_l";
char *YoloDet::OutputMAncBlobName = "output_m";
char *YoloDet::OutputSAncBlobName = "output_s";
int YoloDet::Stride = 32;


YoloDet::YoloDet(const char *planPath, int bs)
{
    // Assign batch size user input as class private member variable
    batchSize = bs;
    // Assign the value of static member to local variable
    int channel = Channel;
    auto newShape = NewShape;
    int outputClasses = OutputClasses;
    int outputFeatures = OutputFeatures;
    // Initialize class private member variables
    modelIn = new float[batchSize * channel * newShape[0] * newShape[1]];
    modelOut = new float[batchSize * outputFeatures * (outputClasses + 5)];
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

    bool flag = initLibNvInferPlugins(&gLogger, "");
    std::cout << "flag: " << flag << std::endl;
    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(modelStream, size);
    context = engine->createExecutionContext();
    std::cout << engine->getNbBindings() << std::endl;
    assert(engine->getNbBindings() == 5);
    inputIndex = engine->getBindingIndex(InputBlobName);
    outputIndex = engine->getBindingIndex(OutputBlobName);
    outputIndex1 = engine->getBindingIndex(OutputBlobName1);
    outputIndex2 = engine->getBindingIndex(OutputBlobName2);
    outputIndex3 = engine->getBindingIndex(OutputBlobName3);



}


Infer::~Infer() {

    //context->destroy();
    //engine->destroy();
    //runtime->destroy();
    delete []modelStream;

}


int Infer::doInfer(const std::vector<cv::Mat*> &images, std::vector<std::vector<DetBox>> &inferOut, int batchSize) {

    // Get the local variable value of NewShape and Stride
    int newShape[2] = {NewShape[0], NewShape[1]};
    int outputFea = OutputFeatures;
    int outputCls = OutputClasses;
    int stride = Stride;
    float modelIn[batchSize * 3 * newShape[0] * newShape[1]];

//    float *modelOut = new float[batchSize * outputFea * (outputCls + 5)];
    float modelOut[batchSize * outputFea * ( outputCls + 5 )];
    std::vector<std::vector<Bbox>> postOut;
    DetBox detBox;
    std::vector<DetBox> boxes;

    // Do pre-process
    preProcess(images, modelIn, newShape, stride, batchSize);

//    std::cout << modelIn[10+200*640] << "," << modelIn[10+200*640+384*640] << "," << modelIn[10+200*640+2*384*640] << std::endl;

    // Pass input data to model and get output data
    detect(modelIn, modelOut, newShape, batchSize, outputFea, outputCls);

//    for (int i = 0; i < sizeof(modelOut)/sizeof(float); i++) {
//        std::cout << i <<":"<< modelOut[i] << std::endl;
//    }

//    for (int i = 0; i < 85; i++) {
//        std::cout << i <<":"<< modelIn[i] << std::endl;
//    }
//    for (int i = 15119*85; i < 15120*85; i++) {
//        std::cout << i <<":"<< modelOut[i] << std::endl;
//    }

    // Do post-process
    int modelOutSize = int(batchSize * outputFea * ( outputCls + 5 ));
    postProcess(images, modelOut, postOut, newShape, modelOutSize);

    for (int i = 0; i < postOut.size(); i++) {
        for (int j = 0; j < postOut[i].size(); j++){
            std::cout << "ClassID" << postOut[i][j].cls << std::endl;
        }
    }

   for (int b = 0; b < batchSize; b++) {
       for (int i = 0; i < postOut[b].size(); i++) {
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


int Infer::scaleFit(const cv::Mat &image, cv::Mat &imagePadding, cv::Scalar color, int newShape[2], int stride) {

    // Get the original shape of image
    int shape[2] = {image.size().width, image.size().height}; // current shape {height, width}
    // Get the minimum fraction ratio
    float ratio = fmin((float) newShape[0] / shape[0], (float) newShape[1] / shape[1]);
    std::cout << newShape[0] << "," << newShape[1] << std::endl;
    // Compute padding {width, height}
    int newUnpad[2] = {(int) round(shape[0] * ratio), (int) round(shape[1] * ratio)};
    std::cout << newUnpad[0] << "," << newUnpad[1] << std::endl;
    int dw = ((newShape[0] - newUnpad[0]) % stride) / 2;
    int dh = ((newShape[1] - newUnpad[1]) % stride) / 2;
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


int Infer::preProcess(const std::vector<cv::Mat*> &images, float *modelIn, int newShape[2], int stride, int batchSize) {

    cv::Scalar color = cv::Scalar(114, 114, 114);
    for (int b = 0; b < batchSize; b++) {
        cv::Mat imagePadding;
        scaleFit(*images[b], imagePadding, color, newShape, stride);
        // Test

        int flag = 0;
        for (int r = 0; r < imagePadding.rows; r++){
            uchar* ucPixel = imagePadding.data + r * imagePadding.step;
            for (int c = 0; c < imagePadding.cols; c++){

                ucPixel += 3;
                flag++;
            }
        }

        int i = 0;
        for (int r = 0; r < imagePadding.rows; r++){
            uchar* ucPixel = imagePadding.data + r * imagePadding.step;
            for (int c = 0; c < imagePadding.cols; c++){
                modelIn[b * 3 * imagePadding.rows * imagePadding.cols + i] = (float)ucPixel[2] / 255.0;
                modelIn[b * 3 * imagePadding.rows * imagePadding.cols + i + imagePadding.rows * imagePadding.cols] = (float)ucPixel[1] / 255.0;
                modelIn[b * 3 * imagePadding.rows * imagePadding.cols + i + 2 * imagePadding.rows * imagePadding.cols] = (float)ucPixel[0] / 255.0;
                ucPixel += 3;
                i++;
            }
        }
    }
    return 0;

}


int Infer::detect(float *modelIn, float *modelOut, int newShape[2], int batchSize,
                  int outputFeatures, int outputClasses) {

    // Run inference
//    auto start = std::chrono::system_clock::now();
    void* buffers[5];

    auto ret1 = cudaMalloc(&buffers[inputIndex], batchSize * 3 * newShape[0] * newShape[1] * sizeof(float));
    auto ret2 = cudaMalloc(&buffers[outputIndex], batchSize * outputFeatures * (outputClasses + 5) * sizeof(float));
    auto ret3 = cudaMalloc(&buffers[outputIndex1], batchSize * 3 * 80 * 48 * (outputClasses + 5) * sizeof(float));
    auto ret4 = cudaMalloc(&buffers[outputIndex2], batchSize * 3 * 40 * 24 * (outputClasses + 5) * sizeof(float));
    auto ret5 = cudaMalloc(&buffers[outputIndex3], batchSize * 3 * 20 * 12 * (outputClasses + 5) * sizeof(float));
    std::cout << "ret1: " << ret1 << std::endl;
    std::cout << "ret2: " << ret2 << std::endl;
    std::cout << "ret3: " << ret3 << std::endl;
    std::cout << "ret4: " << ret4 << std::endl;
    std::cout << "ret5: " << ret5 << std::endl;

    cudaStream_t stream;
    cudaMemcpyAsync(buffers[inputIndex],
                    modelIn,
                    batchSize * 3 * newShape[0] * newShape[1] * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream);
    bool status = context->enqueue(batchSize, buffers, stream, nullptr);
    std::cout << "status: " << status << std::endl;
    if (!status) {
        return 1;
    }
    cudaMemcpyAsync(modelOut,
                    buffers[outputIndex],
                    batchSize * outputFeatures * (outputClasses + 5) * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    cudaFree(buffers[outputIndex1]);
    cudaFree(buffers[outputIndex2]);
    cudaFree(buffers[outputIndex3]);
//    auto end = std::chrono::system_clock::now();

    return 0;

}


int Infer::postProcess(const std::vector<cv::Mat*> &images, float* modelOut, std::vector<std::vector<Bbox>> &postOut,
                       int newShape[2], int modelOutSize, int batchSize ) {

    for (int b = 0; b < batchSize; b++){
        // Init intermediate variables
        std::vector<Bbox> nmsBboxes;
        std::vector<Bbox> scaleBboxes;
        int img0Shape[2] = {images[b]->cols, images[b]->rows};
        //

        nonMaxSuppression(modelOut, nmsBboxes, ConfThres, IouThres, OutputClasses, modelOutSize);
        std::cout << "nmsBoxes size: " << nmsBboxes.size() << std::endl;
        for (int i = 0; i < nmsBboxes.size(); i++) {
            std::cout << "rect: " << nmsBboxes[i].rect << std::endl;
        }

        scaleCoords(newShape, img0Shape, nmsBboxes, scaleBboxes);
        postOut.push_back(scaleBboxes);
    }
    return 0;

}