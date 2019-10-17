#pragma once

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/NativeFunctions.h>

#include <cudnn_v7.h>
#include <vector>
#include <iostream>
#include "utils.cuh"

using c10::IntList;
using at::Tensor;


struct PoolingArgs {
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t idesc, odesc;
  cudnnPoolingDescriptor_t pdesc;

  void setInput(const Tensor& input){
    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&idesc));
    //std::cout << "CudnnCreateTensorDescriptor(idesc), "
    //          << input.size(0) << " " << input.size(1)*input.size(4) << " " << input.size(2) << " " << input.size(3) << std::endl;
    AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(idesc, getCudnnTensorFormat(input), getCudnnDataType(input),
                                              input.size(0), input.size(1)*input.size(4), input.size(2), input.size(3)));
  }
  
  void setOutput(const Tensor& output){
    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&odesc));
    AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(odesc, getCudnnTensorFormat(output), getCudnnDataType(output),
                                              output.size(0), output.size(1)*output.size(4), output.size(2), output.size(3)));
  }

  void setPooling(IntList kernel, IntList pad, IntList stride) {
    AT_CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pdesc));
    AT_CUDNN_CHECK(cudnnSetPooling2dDescriptor(pdesc, CUDNN_POOLING_MAX_DETERMINISTIC, CUDNN_PROPAGATE_NAN,
                                               kernel[0], kernel[1], pad[0], pad[1], stride[0], stride[1]));
  }

  ~PoolingArgs()
  {
    AT_CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pdesc));
    AT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(odesc));
    AT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(idesc));
  }
};


Tensor max_pool2d(
    const Tensor& input_r, IntList kernel, IntList stride, IntList padding) {
  
  auto input = input_r.contiguous();
  //std::cout << "INPUT " << input.sizes() << std::endl;
  //std::cout << "KERNEL " << kernel << std::endl;
  //std::cout << "STRIDE " << stride << std::endl;
  //std::cout << "PADDING " << padding << std::endl;
  auto& ctx = at::globalContext();

  AT_CHECK(ctx.userEnabledCuDNN(), "cuDNN is disabled, but int8 pooling are supported only with cuDNN backend");
  AT_CHECK(input.dim() == 5, "input should have exactly 5 dimensions(NxCxHxWx32(4))");
  AT_CHECK(input.scalar_type() == torch::kInt8, "only int8 data type is supported");
  AT_CHECK((input.size(4) == 4 || input.size(4) == 32), "for int8 dtype number of channels must be exactly divisible by 4 or 32");
  at::native::setCuDNNStreamToCurrent();
  PoolingArgs args;
  //std::cout << "Set input" << std::endl;
  args.setInput(input);
  //std::cout << "Creating pooling descriptor" << std::endl;
  args.setPooling(kernel, padding, stride);
  //std::cout << "Pooling descriptor created" << std::endl;
  args.handle = at::native::getCudnnHandle();
  
  int outB, outC, outH, outW;
  //std::cout << "Computing output dims " << std::endl;
  AT_CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(args.pdesc, args.idesc, &outB, &outC, &outH, &outW));
  //std::cout << outB << " " << outC << " " << outH << " " << outW << std::endl;
  auto output = at::empty({outB, input.size(1), outH, outW, input.size(4)}, input.options());
  //std::cout << "OUTPUT " << output.sizes() << std::endl;
  args.setOutput(output);
  
  float alpha = 1.0f;
  float beta = 0.0f;

  AT_CUDNN_CHECK(cudnnPoolingForward(
    args.handle, args.pdesc,
    &alpha, args.idesc, input.data_ptr(), &beta, args.odesc, output.data_ptr()));
  return output;
}

