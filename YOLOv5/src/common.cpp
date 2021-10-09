//
// Created by shiyi on 2021/9/23.
//

#include "common.h"


int nonMaxSuppression(float *prediction, std::vector<std::vector<float*>> nmsOutput, float confThres, float iouThres,
                      int numClasses, int modelOutSize) {
    // Settings
    int minWH = 2;
    int maxWH = 4096;
    int maxDet = 300;
    int maxNms = 30000;
    float timeLimit = 10.0;
    bool redundant = true;
    std::vector<float> anchorCand;
    std::vector<std::vector<float>> anchorCands;
    // Set clock start
    auto start = std::chrono::system_clock::now();

    // Traverse every anchor box to extract higher object confidence anchor boxes
    for (int i = 4; i < modelOutSize; i += numClasses + 5) {
        if (prediction[i] > confThres) {
            for (int j = i - 4; j < i + numClasses; j++) {
                anchorCand.push_back(prediction[j]);
            }
            anchorCands.push_back(anchorCand);
            anchorCand.clear();
        }
        else {
            continue;
        }
    }

    // Traverse anchor candidates to compute the total confidence, obj_conf * cls_conf
    for (int i = 0; i < anchorCands.size(); i++){
        for (int j = 5; j < numClasses; j++){
            anchorCands[i][j] *= anchorCands[i][4];
        }
    }

}


int cxcywh2xywh(std::vector<std::vector<float>>& bboxes) {
    for (int i = 0; i < bboxes.size(); i++) {
        bboxes[i][0] = bboxes[i][0] - 0.5 * bboxes[i][2];
        bboxes[i][1] = bboxes[i][1] - 0.5 * bboxes[i][3];
    }
}


int nmsBoxes(const std::vector<std::vector<float>>& bboxes, const std::vector<float>& scores,
             const float confThres, const float iouThres, std::vector<int>& indices, const float eta, const int topK) {

}