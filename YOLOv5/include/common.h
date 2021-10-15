//
// Created by shiyi on 2021/9/9.
//

#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <iostream>
#include <vector>
#include <string>
#include <chrono>


#include "opencv2/opencv.hpp"


struct Bbox{
    cv::Rect rect;
    int cls;
    float conf;
};

int nonMaxSuppression(float *prediction, std::vector<Bbox>& nmsBboxes,
                      float confThres, float iouThres, int numClasses);

int cxcywh2xywh(std::vector<float>& boxes);

bool descendingSort(const Bbox& a, const Bbox& b);

int scaleCoords(const int imgShape[2], const int img0Shape[2], std::vector<Bbox> &nmsBboxes, bool ratioPad = false);


#endif //YOLOV5_COMMON_H_
