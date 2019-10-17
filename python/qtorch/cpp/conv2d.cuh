#pragma once

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cudnn/Handle.h>

#include <iostream>
#include <string>

#include <cudnn_v7.h>
#include <vector>
#include "utils.cuh"

using c10::IntList;
using at::Tensor;
using std::string;
constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int weight_output_channels_dim = 0;


static std::vector<int64_t> conv_output_size(
    std::vector<int64_t> input_size, std::vector<int64_t> weight_size, IntList padding, IntList stride, IntList dilation) 
{
  auto dims = input_size.size();
  
  auto vect_c = input_size[dims-1];

  std::vector<int64_t> output_size(dims);                            // usual shapes + vect_c
  output_size[0] = input_size[input_batch_size_dim];                 // batch
  output_size[1] = weight_size[weight_output_channels_dim] / vect_c; // channels
  output_size[dims-1] = vect_c;                                      // vect_c

  for (size_t d = 2; d < dims - 1; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}


struct ConvolutionArgs {
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t idesc, odesc;
  cudnnFilterDescriptor_t wdesc;
  const Tensor& input, output, weight;
  at::native::ConvolutionDescriptor cdesc;

  ConvolutionArgs(const Tensor& input, const Tensor& output, const Tensor& weight, const string& config) : input(input), output(output), weight(weight) {
    auto tensorFormat = getCudnnTensorFormat(input);
    cudnnDataType_t dataType = getCudnnDataType(input);
    AT_CHECK(dataType == getCudnnDataType(weight) && tensorFormat == getCudnnTensorFormat(weight))

    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&idesc));
    AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(idesc, tensorFormat, dataType, 
                                              input.size(0), input.size(1)*input.size(4), input.size(2), input.size(3)));      // N,C*VECT_C,H,W

    AT_CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc));
    AT_CUDNN_CHECK(cudnnSetFilter4dDescriptor(wdesc, dataType, tensorFormat, 
                                              weight.size(0), weight.size(1)*weight.size(4), weight.size(2), weight.size(3))); // N,C*VECT_C,H,W

    std::vector<int64_t> output_size = {};
    if (config == "external")
    {
      tensorFormat = CUDNN_TENSOR_NCHW;
      dataType = CUDNN_DATA_FLOAT;
      output_size = {output.size(0), output.size(1), output.size(2), output.size(3)};
    }
    else
      output_size = {output.size(0), output.size(1) * output.size(4), output.size(2), output.size(3)};

    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&odesc));
    AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(odesc, tensorFormat, dataType, 
                                              output.size(0), output.size(1), output.size(2), output.size(3))); // N,C*VECT_C,H,W
  }
  
  void setConv(IntList pad, IntList stride, IntList upscale /* aka dilation */, int groups=1) {
    AT_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(cdesc.mut_desc(), pad[0], pad[1], stride[0], stride[1], upscale[0], upscale[1], 
                                                   CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32));
    AT_CUDNN_CHECK(cudnnSetConvolutionGroupCount(cdesc.mut_desc(), groups));
    AT_CUDNN_CHECK(cudnnSetConvolutionMathType(cdesc.mut_desc(), CUDNN_TENSOR_OP_MATH));
  }

  ~ConvolutionArgs()
  {
    AT_CUDNN_CHECK(cudnnDestroyFilterDescriptor(wdesc));
    AT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(odesc));
    AT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(idesc));
  }
};


Tensor conv2d(
    const Tensor& input_r, const Tensor& weight_r, const string& config, double scale, IntList stride, IntList padding, IntList dilation, int groups) 
{  
  auto input = input_r.contiguous();
  auto weight = weight_r.contiguous();
  
  auto& ctx = at::globalContext();
  AT_CHECK(ctx.userEnabledCuDNN(), "cuDNN is disabled, but int8 convolutions are supported only with cuDNN backend");
  AT_CHECK(weight.dim() == 5, "weight should have exactly 5 dimensions, NxC/4xHxWx4 or NxC/32xHxWx32");
  AT_CHECK(input.dim() == 5, "input should have exactly 5 dimensions, NxC/4xHxWx4 or NxC/32xHxWx32");
  AT_CHECK((input.size(4) == 4 && weight.size(0) % 4 == 0 && weight.size(4) == 4) || 
           (input.size(4) == 32 && weight.size(0) % 32 == 0 && weight.size(4) == 32))
  at::native::setCuDNNStreamToCurrent();

  AT_CHECK(config == "native" || (config == "external" && input.size(4) == 4));

  at::TensorArg input_args = {input, "input", 0};
  at::TensorArg weight_args = {weight, "weight", 0};

  at::CheckedFrom c = "cudnn_int8_convolution";
  at::checkAllSameType(c, {input_args, weight_args});
  at::checkAllSameGPU(c, {input_args, weight_args});
  
  std::vector<int64_t> input_size = {input.size(0), input.size(1), input.size(2), input.size(3), input.size(4)};
  std::vector<int64_t> weight_size = {weight.size(0), weight.size(1), weight.size(2), weight.size(3), weight.size(4)};
  auto output_size = conv_output_size(input_size, weight_size, padding, stride, dilation);


  auto dtype = torch::kInt8;
  if (config == "external")
  {
    output_size[1] *= output_size[4];
    output_size.pop_back();
    dtype = at::kFloat;
  }

  auto output = at::empty(
                    output_size,
                    at::TensorOptions().dtype(dtype)
                                       .device(input.device())
                                       .layout(input.layout())
                                       .is_variable(input.is_variable()));
  ConvolutionArgs args{ input, output, weight, config };
  //std::clog << "Output size " << output.sizes() << std::endl;
  args.handle = at::native::getCudnnHandle();
  args.setConv(padding, stride, dilation);
  
  cudnnConvolutionFwdAlgo_t fwdAlg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  size_t workspaceSizeInBytes;
  cudnnGetConvolutionForwardWorkspaceSize(args.handle, args.idesc, args.wdesc, args.cdesc.mut_desc(), args.odesc, fwdAlg, &workspaceSizeInBytes);
  auto workspace = at::empty({(signed long)workspaceSizeInBytes}, at::device(output.device()).dtype(torch::kInt8));

  float beta = 0.0f;
  float alpha = scale;

  AT_CUDNN_CHECK(cudnnConvolutionForward(
    args.handle,
    &alpha, args.idesc, input.data_ptr(),
    args.wdesc, weight.data_ptr(),
    args.cdesc.mut_desc(), fwdAlg, workspace.data_ptr(), workspaceSizeInBytes,
    &beta, args.odesc, output.data_ptr()));
  return output;
}
