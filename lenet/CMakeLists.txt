cmake_minimum_required(VERSION 3.19)
project(lenet)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-O2 -pthread)

include_directories(include)
# cuda
include_directories(/usr/local/cuda-11.3/include)
link_directories(/usr/local/cuda-11.3/lib64)
# tensorrt
include_directories(/home/shiyi/3rdParty/TensorRT-8.0.1.6/include)
link_directories(/home/shiyi/3rdParty/TensorRT-8.0.1.6/lib)
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(lenet src/lenet.cpp src/ReadMnist.cpp)
target_link_libraries(lenet nvinfer)
target_link_libraries(lenet nvonnxparser)
target_link_libraries(lenet cudart)
target_link_libraries(lenet ${OpenCV_LIBS})
