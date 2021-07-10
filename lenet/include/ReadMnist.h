//
// Created by shiyi on 2021/7/9.
//

#ifndef LENET_INCLUDE_READMNIST_H_
#define LENET_INCLUDE_READMNIST_H_

#include <opencv2/opencv.hpp>

class ReadMnist
{
 public:
  cv::Mat readImg(std::string&);

  cv::Mat readLabel(std::string&);

 private:
  int cvtCharArrayToInt(unsigned char*, int);

  bool isImgDataFile(unsigned char*, int);

  bool isLabelDataFile(unsigned char*, int);

  cv::Mat readData(std::fstream&, int, int);

  cv::Mat readImgData(std::fstream&, int);

  cv::Mat readLabelData(std::fstream&, int);
};

#endif //LENET_INCLUDE_READMNIST_H_
