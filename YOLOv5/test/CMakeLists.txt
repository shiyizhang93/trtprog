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

## lib
#link_directories(${PROJECT_SOURCE_DIR}/lib)
#
#set(SRC_LIST src/infer.cpp src/common.cpp)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#add_library(yolov5 SHARED ${SRC_LIST})

add_executable(test src/test.cpp)

target_link_libraries(test nvinfer)
target_link_libraries(test nvonnxparser)
target_link_libraries(test cudart)
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(test nvinfer_plugin)