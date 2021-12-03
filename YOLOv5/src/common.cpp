//
// Created by shiyi on 2021/9/23.
//

#include <algorithm>
#include "common.h"


int nonMaxSuppression(float *prediction, std::vector<Bbox> &nmsBboxes, float confThres, float iouThres,
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
    Bbox bbox;
    std::vector<Bbox> bboxes;
    std::vector<cv::Rect> rects;
    cv::Rect rect;
    std::vector<float> score;
    Bbox nmsBbox;

    // Set clock start
    auto start = std::chrono::system_clock::now();

    // Traverse every anchor box to extract higher object confidence anchor boxes
    for (int i = 4; i < modelOutSize; i += (numClasses + 5) ) {

        if (prediction[i] > confThres) {
            for (int j = i - 4; j < ((i - 4) + numClasses + 5); j++) {
                anchorCand.push_back(prediction[j]);
            }
            anchorCands.push_back(anchorCand);
            anchorCand.clear();
        }
        else {
            continue;
        }
    }

//    for(int i = 0; i < 85; i++) {
//        std::cout << i << ":" << anchorCands[34][i] << std::endl;
//    }

    // Check the number of anchors
    if (anchorCands.size() == 0) {
        return 1;
    }

    // Traverse anchor candidates to compute the total confidence, obj_conf * cls_conf
    for (int i = 0; i < anchorCands.size(); i++) {
        for (int j = 5; j < (numClasses+5); j++){
            anchorCands[i][j] *= anchorCands[i][4];
        }
    }

//    for(int i = 0; i < 85; i++) {
//        std::cout << i << ":" << anchorCands[0][i] << std::endl;
//    }

    // Filter best class only
    // bboxes(cv::Rect, cls, conf)
    for (int i = 0; i < anchorCands.size(); i++) {
        // anchor candidates (center x, center y, width, height) to (x1, y1, width, height)
        cxcywh2xywh(anchorCands[i]);
        // convert std::vector<float> to cv::Rect and copy the result to bboxes
        bbox.rect = cv::Rect(anchorCands[i][0],
                             anchorCands[i][1],
                             anchorCands[i][2],
                             anchorCands[i][3]);
        // Find the max confidence value and the corresponding value position
        bbox.cls = std::max_element(anchorCands[i].begin() + 5, anchorCands[i].end()) - (anchorCands[i].begin() + 5);
        bbox.conf = *std::max_element(anchorCands[i].begin()+5, anchorCands[i].end());
        bboxes.push_back(bbox);
    }
    // Check if the size of bboxes greater than maxNms, descending Sort the bboxes and erase the excess bboxes
    if (bboxes.size() > maxNms) {
        std::cout << "Exceeding max size of the number of bounding boxes ..." << std::endl;
        std::sort(bboxes.begin(), bboxes.end(), descendingSort);
        for (int i = 0; i < bboxes.size() - maxNms; i++){
            bboxes.erase(bboxes.begin() + maxNms + i);
        }
    }

    std::vector<int> classID;
    // Batched NMS
    // rect offset by class, and conf
    for (int i = 0; i < bboxes.size(); i++) {
        rect = cv::Rect(bboxes[i].rect.x + bboxes[i].cls * maxWH,
                        bboxes[i].rect.y + bboxes[i].cls * maxWH,
                        bboxes[i].rect.width,
                        bboxes[i].rect.height);
        rects.push_back(rect);
        score.push_back(bboxes[i].conf);
        classID.push_back(bboxes[i].cls);
    }
    std::cout << "rect_size: " << rects.size() << std::endl;
    for (int i = 0; i < score.size(); i++) {
        std::cout << classID[i] << ": " << score[i] << std::endl;
        std::cout << rects[i] << std::endl;
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(rects, score, confThres, iouThres, indices);
    std::cout << "indices size: " << indices.size() << std::endl;
    // Get the left boxes
    if (indices.size() > maxDet) {
        for (int i = 0; i < maxDet; i++) {
            int idx = indices[i];
            nmsBbox.rect = bboxes[idx].rect;
            nmsBbox.cls = bboxes[idx].cls;
            nmsBbox.conf = bboxes[idx].conf;

            nmsBboxes.push_back(nmsBbox);
        }
    }
    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        nmsBbox.rect = bboxes[idx].rect;
        nmsBbox.cls = bboxes[idx].cls;
        nmsBbox.conf = bboxes[idx].conf;

        nmsBboxes.push_back(nmsBbox);
    }

    return 0;
}


int cxcywh2xywh(std::vector<float>& boxes) {

    boxes[0] = boxes[0] - 0.5 * boxes[2];
    boxes[1] = boxes[1] - 0.5 * boxes[3];

    return 0;
}


bool descendingSort(const Bbox& a, const Bbox& b) {

        return a.conf > b.conf;
    }


int scaleCoords(const int imgShape[2], const int img0Shape[2], std::vector<Bbox> &nmsBboxes,
                std::vector<Bbox> &scaleBboxes, bool ratioPad) {
    // Rescale coords (xywh) from imgShape to img0Shape
    float gain = fmin(float(imgShape[0]) / float(img0Shape[0]), float(imgShape[1]) / float(img0Shape[1]));
    int pad[2] = {static_cast<int>((imgShape[1] - img0Shape[1] * gain) / 2), static_cast<int>((imgShape[0] - img0Shape[0] * gain) / 2)};
    for (int i = 0; i < nmsBboxes.size(); i ++) {
        Bbox box;
        box.rect.x = (nmsBboxes[i].rect.x - pad[1]) / gain;
        box.rect.y = (nmsBboxes[i].rect.y - pad[0]) / gain;
        box.rect.width = nmsBboxes[i].rect.width / gain;
        box.rect.height = nmsBboxes[i].rect.height / gain;
        box.cls = nmsBboxes[i].cls;
        box.conf = nmsBboxes[i].conf;

        std::cout << "rect: " <<box.rect << std::endl;


        int flag = clipCoords(box.rect, img0Shape);
        if (flag == 1){
            continue;
//            scaleBboxes.erase(scaleBboxes.begin() + i);
        }
        else {
            scaleBboxes.push_back(box);
        }
    }
}


int clipCoords(cv::Rect &box, const int imgShape[2]) {
    if (box.x < 0 ) {
        box.x = 0;
    }
    else if (box.x > imgShape[0]) {
        return 1;
    }
    //
    if (box.y < 0) {
        box.y = 0;
    }
    else if (box.y > imgShape[1]) {
        return 1;
    }
    //
    if ((box.x + box.width) < 0) {
        return 1;
    }
    else if ((box.x + box.width) > imgShape[0]) {
        box.width = imgShape[0] - box.x;
    }
    //
    if ((box.y + box.height) < 0) {
        return 1;
    }
    else if ((box.y + box.height) > imgShape[1]) {
        box.height = imgShape[1] - box.y;
    }

    return 0;
}