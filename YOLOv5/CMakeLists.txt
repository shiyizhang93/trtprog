# Set cmake version
cmake_minimum_required(VERSION 3.19)

# Set project name and version
project(yolov5Det VERSION 0.4)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-O2 -debug -pthread)

include_directories(include)

# Configure a header file to pass the version number to the source code
configure_file(include/yolov5DetConfig.h.in yolov5DetConfig.h)

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

add_executable(yolov5Det src/main.cpp src/yolov5Det.cpp src/common.cpp include/common.h)

target_link_libraries(yolov5Det nvinfer)
target_link_libraries(yolov5Det nvonnxparser)
target_link_libraries(yolov5Det cudart)
target_link_libraries(yolov5Det ${OpenCV_LIBS})

target_include_directories(yolov5Det PUBLIC "${PROJECT_BINARY_DIR}")
