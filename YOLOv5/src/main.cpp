//
// Created by shiyi on 2021/10/21.
//

#include <chrono>
#include "opencv2/opencv.hpp"
#include "infer.h"


int main() {

    const char *PlanPath = "model.plan";
    std::string inputImg = "test.jpg";
    cv::Mat img = cv::imread(inputImg);
    std::vector<cv::Mat*> images;
    images.push_back(&img);
    std::vector<std::vector<DetBox>> outBoxes;

    Infer yolov5Det(PlanPath);

    yolov5Det.doInfer(images, outBoxes);

    for (int i = 0; i < (int) images.size(); i++)
    {
        for (int j = 0; j < (int) outBoxes[i].size(); j++)
        {
            std::cout << "j: " << j << std::endl;
            std::cout << outBoxes[i][j].classId << std::endl;
//            if (outBoxes[i][j].classId == 0)
//            {
            cv::rectangle(*images[i],
                          cv::Point(outBoxes[i][j].x,outBoxes[i][j].y),
                          cv::Point(outBoxes[i][j].x + outBoxes[i][j].width, outBoxes[i][j].y + outBoxes[i][j].height),
                          cv::Scalar (0, 0, 255), 2);
//            }
//            else if (outBoxes[i][j].classId == 1)
//            {
//                cv::rectangle(*images[i],
//                              cv::Point(outBoxes[i][j].x,outBoxes[i][j].y),
//                              cv::Point(outBoxes[i][j].x + outBoxes[i][j].width, outBoxes[i][j].y + outBoxes[i][j].height),
//                              cv::Scalar (255, 0, 0), 2);
//            }
//
            cv::putText(*images[i],
                        "ClassID: " + std::to_string(outBoxes[i][j].classId),
                        cv::Point(outBoxes[i][j].x, outBoxes[i][j].y),
                        cv::FONT_HERSHEY_TRIPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 1);
        }
        cv::imwrite("./"+ std::to_string(i) + ".jpg", *images[i]);
    }

}