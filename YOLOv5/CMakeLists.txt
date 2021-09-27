cmake_minimum_required(VERSION 3.19)
project(yolov5)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-O2 -pthread)

include_directories(include)
# cuda
include_directories(/usr/local/cuda-11.3/include)
link_directories(/usr/local/cuda-11.3/lib64)
# tensorrt
include_directories(/home/shiyi/3rdParty/TensorRT-8.0.3.4/include)
link_directories(/home/shiyi/3rdParty/TensorRT-8.0.3.4/lib)
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(yolov5 src/infer.cpp src/common.cpp)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 nvonnxparser)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 ${OpenCV_LIBS})