cmake_minimum_required(VERSION 3.26.3)

project(LOADHDF5)

find_package(HighFive REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} HighFive)