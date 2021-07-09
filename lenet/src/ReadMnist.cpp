//
// Created by shiyi on 2021/7/9.
//

#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "ReadMnist.h"

using namespace std;

const int MAGICNUMIMG = 2051;
const int MAGICNUMLABEL = 2049;
const int IMG_H = 28;
const int IMG_W = 28;


struct mnistImgTmp
{
  unsigned char magic_num[4];
  unsigned char num_img[4];
  unsigned char num_rows[4];
  unsigned char num_col[4];
};

struct mnistLabelTmp
{
  unsigned char magic_num[4];
  unsigned char num_labels[4];
};


int ReadMnist::cvtCharArrayToInt(unsigned char* array, int lenth_of_array)
{
  if (lenth_of_array < 0)
    return -1;
  int result = static_cast<signed int>(array[0]);
  for (int i = 1; i < lenth_of_array; i++)
    result = (result << 8) + array[i];

  return result;
}


bool ReadMnist::isImgDataFile(unsigned char* magic_num, int length_array)
{
  int magic_num_img = cvtCharArrayToInt(magic_num, length_array);
  if (magic_num_img == MAGICNUMIMG)
    return true;

  return false;
}


bool ReadMnist::isLabelDataFile(unsigned char* magic_num, int length_array)
{
  int magic_num_label = cvtCharArrayToInt(magic_num, length_array);
  if(magic_num_label == MAGICNUMLABEL)
    return true;

  return false;
}


cv::Mat ReadMnist::readData(std::fstream& data_file, int number_data, int data_size_bytes)
{
  cv::Mat data_mat;
  if (data_file.is_open())
  {
    int all_data_size_bytes = data_size_bytes * number_data;
    unsigned char* tmp_data = new unsigned char[all_data_size_bytes];
    data_file.read((char*)tmp_data, all_data_size_bytes);
    data_mat = cv::Mat(number_data, data_size_bytes, CV_8UC1, tmp_data).clone();
    delete [] tmp_data;
    data_file.close();
  }

  return data_mat;
}


cv::Mat ReadMnist::readImgData(std::fstream& img_data_file, int num_img)
{
  int img_size_bytes = 28 * 28;

  return readData(img_data_file, num_img, img_size_bytes);
}


cv::Mat ReadMnist::readLabelData(std::fstream& label_data_file, int num_label)
{
  int label_size_bytes = 1;

  return readData(label_data_file, num_label, label_size_bytes);
}


cv::Mat ReadMnist::readImg(std::string& filename)
{
  std::fstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
  if (!file.is_open())
    return cv::Mat();
  mnistImgTmp file_img;
  file.read((char *)(&file_img), sizeof(file_img));
  if (!isImgDataFile(file_img.magic_num, 4))
    return cv::Mat();
  int number_img = cvtCharArrayToInt(file_img.num_img, 4);

  return readImgData(file, number_img);
}


cv::Mat ReadMnist::readLabel(std::string& filename)
{
  std::fstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
  if (!file.is_open())
    return cv::Mat();
  mnistLabelTmp file_label;
  file.read((char *)(&file_label), sizeof(file_label));
  if (!isLabelDataFile(file_label.magic_num, 4))
    return cv::Mat();
  int number_label = cvtCharArrayToInt(file_label.num_labels, 4);

  return readLabelData(file, number_label);
}
