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

int cxcywh2xywh(std::vector<float*>& box);

int nmsBoxes(const std::vector<std::vector<float>>& bboxes, const std::vector<float>& scores,
             const float confThres, const float iouThres, std::vector<int>& indices,
             const float eta = 1.f, const int topK = 0);

bool cmp();

float iou();


#endif //YOLOV5_COMMON_H_
