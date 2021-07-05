cmake_minimum_required(VERSION 3.19)
project(lenet)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-O2 -pthread)

include_directories(include)
# cuda
include_directories(/usr/local/cuda-11.2/include)
link_directories(/usr/local/cuda-11.2/lib64)
# tensorrt
include_directories(/home/shiyi/3rdparty/TensorRT-7.2.3.4/include)
link_directories(/home/shiyi/3rdparty/TensorRT-7.2.3.4/lib)

add_executable(lenet src/lenet.cpp)
target_link_libraries(lenet nvinfer)
target_link_libraries(lenet cudart)