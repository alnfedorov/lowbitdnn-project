#ifndef LIBLOWBIT_LIBCONVBENCHMARK_H
#define LIBLOWBIT_LIBCONVBENCHMARK_H

#include <chrono>
#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <map>
#include <cudnn.h>
#include <cuda_device_runtime_api.h>

#include "half.hpp"
using half_float::half;

const std::map<std::string, cudnnTensorFormat_t> tensorFormatMapping = {
    {"NCHW", CUDNN_TENSOR_NCHW},
    {"NHWC", CUDNN_TENSOR_NHWC},
    {"NCHW_VECT_C", CUDNN_TENSOR_NCHW_VECT_C}
};

const std::map<std::string, cudnnDataType_t> dataTypeMapping = {
    {"float", CUDNN_DATA_FLOAT},
    {"double", CUDNN_DATA_DOUBLE},
    {"half", CUDNN_DATA_HALF},
    {"int8", CUDNN_DATA_INT8},
    {"int32", CUDNN_DATA_INT32},
    {"int8x4", CUDNN_DATA_INT8x4},
    {"uint8", CUDNN_DATA_UINT8},
    {"uint8x4", CUDNN_DATA_UINT8x4},
    {"int8x32", CUDNN_DATA_INT8x32}
};

template <typename InputDataType, typename FilterDataType, typename OutDataType>
std::chrono::microseconds benchmark_convolution(size_t B, size_t C, size_t H, size_t W,
                                                size_t numFilters, size_t filterH, size_t filterW,
                                                size_t padH, size_t padW, size_t strideH, size_t strideW, size_t dilationH, size_t dilationW,
                                                cudnnTensorFormat_t inputTensorFormat, cudnnTensorFormat_t filterTensorFormat, cudnnTensorFormat_t outputTensorFormat, 
                                                cudnnDataType_t inputDataType, cudnnDataType_t filterDataType, 
                                                cudnnDataType_t convAccumulatorDataType, cudnnDataType_t outDataType,
                                                int verbose=1);

// FLOAT CONFIG
extern template std::chrono::microseconds benchmark_convolution<float, float, float>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// TRUE_HALF CONFIG
extern template std::chrono::microseconds benchmark_convolution<half, half, half>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t,
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// DOUBLE CONFIG
extern template std::chrono::microseconds benchmark_convolution<double, double, double>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// INT8* CONFIG
extern template std::chrono::microseconds benchmark_convolution<int8_t, int8_t, int8_t>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// INT8*_EXT CONFIG
extern template std::chrono::microseconds benchmark_convolution<int8_t, int8_t, float>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// UINT8* CONFIG
extern template std::chrono::microseconds benchmark_convolution<uint8_t, int8_t, int8_t>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);
// UINT8*_EXT CONFIG
extern template std::chrono::microseconds benchmark_convolution<uint8_t, int8_t, float>(
    size_t, size_t, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, size_t, size_t, size_t,
    cudnnTensorFormat_t, cudnnTensorFormat_t, cudnnTensorFormat_t, 
    cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, cudnnDataType_t, int);


#endif //LIBLOWBIT_LIBCONVBENCHMARK_H
