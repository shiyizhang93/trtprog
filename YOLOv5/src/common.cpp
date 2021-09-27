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

    // Traverse anchor candidates to compute the total confidence - obj_conf * cls_conf
    for (int i = 0; i < anchorCands.size(); i++){
        for (int j = 5; j < numClasses; j++){
            anchorCands[i][j] *= anchorCands[i][4];
        }
    }


}


int xywh2xyxy(std::vector<float*>& box){

}