cmake_minimum_required(VERSION 3.19)
project(yolov5)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-O2 -debug -pthread)

include_directories(include)
# cuda
include_directories(/usr/local/cuda-11.5/include)
link_directories(/usr/local/cuda-11.5/lib64)
# tensorrt
include_directories(/home/shiyi/3rdParty/TensorRT-8.2.1.8/include)
link_directories(/home/shiyi/3rdParty/TensorRT-8.2.1.8/lib)
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

## lib
#link_directories(${PROJECT_SOURCE_DIR}/lib)
#
#set(SRC_LIST src/infer.cpp src/common.cpp)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#add_library(yolov5 SHARED ${SRC_LIST})

add_executable(yolov5det src/main.cpp src/yoloDet.cpp src/common.cpp include/common.h)

target_link_libraries(yolov5det nvinfer)
target_link_libraries(yolov5det nvonnxparser)
target_link_libraries(yolov5det cudart)
target_link_libraries(yolov5det ${OpenCV_LIBS})
