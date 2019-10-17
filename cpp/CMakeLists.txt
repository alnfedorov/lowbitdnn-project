cmake_minimum_required(VERSION 3.1...3.14)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(liblowbit-cnn VERSION 0.1 LANGUAGES CXX CUDA)

set(CUDA_ARCH "75" CACHE STRING "CUDA target architecture (PPTX and SSAS in the same time). Examples are 30, 75, 72 etc")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)


if ($ENV{CLION_IDE})
    include_directories("/usr/local/cuda/include")
endif ()


# TODO move everything to the separate third_party folder
# BOOST
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45 REQUIRED)
# Torch
include_directories("/usr/include/python3.7m/")
set(ENV{Torch_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch")
find_package(Torch REQUIRED)
# OpenMP
find_package(OpenMP REQUIRED)
#
add_subdirectory(int8conv)
#add_subdirectory(libbenchmark)
#add_subdirectory(apps)