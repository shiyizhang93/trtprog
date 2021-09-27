//
// Created by shiyi on 2021/9/9.
//

#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <iostream>
#include <vector>
#include <string>
#include <chrono>


int nonMaxSuppression(float *prediction, std::vector<std::vector<float*>> nmsOutput,
                      float confThres, float iouThres, int numClasses);

int xywh2xyxy(std::vector<float*>& box);

bool cmp();

float iou();


#endif //YOLOV5_COMMON_H_
