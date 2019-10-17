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
constexpr int input_batch_size_dim2 = 0;  // also grad_input
constexpr int weight_output_channels_dim2 = 0;


static std::vector<int64_t> conv_output_size_backward(
    std::vector<int64_t> input_size, std::vector<int64_t> weight_size, IntList padding, IntList stride, IntList dilation) 
{
  auto dims = input_size.size();

  std::vector<int64_t> output_size(dims);                            // usual shapes
  output_size[0] = input_size[input_batch_size_dim2];                 // batch
  output_size[1] = weight_size[weight_output_channels_dim2];          // channels

  for (size_t d = 2; d < dims; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}


struct ConvolutionBackFilterArgs {
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t gdesc, idesc;
  cudnnFilterDescriptor_t fdesc;
  const Tensor& input, gradient, filters;
  at::native::ConvolutionDescriptor cdesc;

  ConvolutionBackFilterArgs(const Tensor& input, const Tensor& gradient, const Tensor& filters) : input(input), gradient(gradient), filters(filters) {
    auto tensorFormat = CUDNN_TENSOR_NCHW;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&idesc));
    AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(idesc, tensorFormat, dataType, 
                                              input.size(0), input.size(1), input.size(2), input.size(3)));      // N,C,H,W

    AT_CUDNN_CHECK(cudnnCreateFilterDescriptor(&fdesc));
    AT_CUDNN_CHECK(cudnnSetFilter4dDescriptor(fdesc, dataType, tensorFormat, 
                                              filters.size(0), filters.size(1), filters.size(2), filters.size(3))); // N,C,H,W

    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&gdesc));
    AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(gdesc, tensorFormat, dataType, 
                                              gradient.size(0), gradient.size(1), gradient.size(2), gradient.size(3))); // N,C,H,W
  }
  
  void setConv(IntList stride, IntList pad, IntList upscale /* aka dilation */, int groups=1) {
    AT_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(cdesc.mut_desc(), pad[0], pad[1], stride[0], stride[1], upscale[0], upscale[1], 
                                                   CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    AT_CUDNN_CHECK(cudnnSetConvolutionGroupCount(cdesc.mut_desc(), groups));
  }

  ~ConvolutionBackFilterArgs()
  {
    AT_CUDNN_CHECK(cudnnDestroyFilterDescriptor(fdesc));
    AT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(idesc));
    AT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(gdesc));
  }
};


Tensor conv2d_backward_data(
    const Tensor& input_r, const Tensor& grads_r, IntList stride, IntList padding, IntList dilation, int groups) 
{  
  AT_ASSERT(groups == 1);
  auto input = input_r.contiguous();
  auto grad = grads_r.contiguous();
  
  auto& ctx = at::globalContext();
  AT_CHECK(ctx.userEnabledCuDNN(), "cuDNN is disabled, but int8 convolutions are supported only with cuDNN backend");
  at::native::setCuDNNStreamToCurrent();

  at::TensorArg input_args = {input, "input", 0};
  at::TensorArg grad_args = {grad, "gradients", 0};

  at::CheckedFrom c = "cudnn conv backward";
  at::checkAllSameType(c, {input_args, grad_args});
  at::checkAllSameGPU(c, {input_args, grad_args});
  
//   std::vector<int64_t> input_size = {input.size(0), input.size(1), input.size(2), input.size(3)};
//   std::vector<int64_t> grad_size = {grad.size(0), grad.size(1), grad.size(2), grad.size(3)};
//   auto output_size = conv_output_size_backward(input_size, grad_size, padding, stride, dilation);
  std::vector<int64_t> output_size = {grad.size(1), input.size(1), 3, 3};

  auto filters = at::empty(output_size, grad.options());
  ConvolutionBackFilterArgs args{ input, grad, filters};
  //std::clog << "Output size " << filters.sizes() << std::endl;
  args.handle = at::native::getCudnnHandle();
  args.setConv(padding, stride, dilation);
  
  cudnnConvolutionBwdFilterAlgo_t algo;
  cudnnGetConvolutionBackwardFilterAlgorithm(args.handle, args.idesc, args.gdesc, args.cdesc.mut_desc(), 
                                                                                   args.fdesc, CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &algo);

  size_t workspaceSizeInBytes;
  cudnnGetConvolutionBackwardFilterWorkspaceSize(args.handle, args.idesc, args.gdesc, args.cdesc.mut_desc(), args.fdesc, algo, &workspaceSizeInBytes);
  auto workspace = at::empty({(signed long)workspaceSizeInBytes}, at::device(grad.device()).dtype(torch::kFloat32));

  float alpha = 1.0f;
  float beta = 0.0f;

  AT_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
    args.handle,
    &alpha, args.idesc, input.data_ptr(),
    args.gdesc, grad.data_ptr(),
    args.cdesc.mut_desc(), algo, workspace.data_ptr(), workspaceSizeInBytes,
    &beta, args.fdesc, filters.data_ptr()));
  return filters;
}
