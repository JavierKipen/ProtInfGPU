cmake_minimum_required(VERSION 3.26.3)

project(LOADHDF5)

enable_language("CUDA")

set(CMAKE_PREFIX_PATH third_party/HighFive/build/install/)

find_package(HighFive REQUIRED)

add_executable(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME} HighFive -lcublas)