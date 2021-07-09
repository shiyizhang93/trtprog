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
# OpenCV
find_package(OpenCV 4.5.2 REQUIRED)

add_executable(lenet src/lenet.cpp src/ReadMnist.cpp)
#add_executable(lenet src/lenet_bk.cpp)
target_link_libraries(lenet nvinfer)
target_link_libraries(lenet nvonnxparser)
target_link_libraries(lenet cudart)
target_link_libraries(lenet ${OpenCV_LIBS})