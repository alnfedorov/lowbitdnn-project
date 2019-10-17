#pragma once

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cudnn/Handle.h>

#include <iostream>
#include <string>
#include <tuple>

#include <cudnn_v7.h>
#include <vector>


using c10::IntList;
using at::Tensor;
using std::string;
constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int weight_output_channels_dim = 0;


static std::vector<int64_t> conv_output_size(
    std::vector<int64_t> input_size, std::vector<int64_t> weight_size, IntList padding, IntList stride, IntList dilation) 
{
  auto dims = input_size.size();

  std::vector<int64_t> output_size(dims);                            // usual shapes + vect_c
  output_size[0] = input_size[input_batch_size_dim];                 // batch
  output_size[1] = weight_size[weight_output_channels_dim];          // channels

  for (size_t d = 2; d < dims; ++d) {
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

  ConvolutionArgs(const Tensor& input, const Tensor& output, const Tensor& weight, 
                  cudnnTensorFormat_t tensorFormat, cudnnDataType_t dataType) : input(input), output(output), weight(weight) {

    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&idesc));
    AT_CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc));
    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&odesc));

    if (tensorFormat == CUDNN_TENSOR_NCHW_VECT_C)
    {
      AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(idesc, tensorFormat, dataType, 
                                              input.size(0), input.size(1) * 32, input.size(2), input.size(3)));

      AT_CUDNN_CHECK(cudnnSetFilter4dDescriptor(wdesc, dataType, tensorFormat, 
                                              weight.size(0), weight.size(1) * 32, weight.size(2), weight.size(3)));

      AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(odesc, tensorFormat, dataType, 
                                              output.size(0), output.size(1) * 32, output.size(2), output.size(3)));
    }
    else
    {
      AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(idesc, tensorFormat, dataType, 
                                              input.size(0), input.size(1), input.size(2), input.size(3)));

      AT_CUDNN_CHECK(cudnnSetFilter4dDescriptor(wdesc, dataType, tensorFormat, 
                                              weight.size(0), weight.size(1), weight.size(2), weight.size(3)));

      AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(odesc, tensorFormat, dataType, 
                                              output.size(0), output.size(1), output.size(2), output.size(3)));
    }
  }
  
  void setConv(IntList pad, IntList stride, IntList upscale /* aka dilation */, int groups, cudnnDataType_t accumulator) {
    AT_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(cdesc.mut_desc(), pad[0], pad[1], stride[0], stride[1], upscale[0], upscale[1], 
                                                   CUDNN_CROSS_CORRELATION, accumulator));
    AT_CUDNN_CHECK(cudnnSetConvolutionGroupCount(cdesc.mut_desc(), groups));
    AT_CUDNN_CHECK(cudnnSetConvolutionMathType(cdesc.mut_desc(), CUDNN_DEFAULT_MATH));
  }

  ~ConvolutionArgs()
  {
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyTensorDescriptor(odesc);
    cudnnDestroyTensorDescriptor(idesc);
  }
};


std::tuple<Tensor, float> cudnnConv2DFloat(
    const Tensor& input_r, const Tensor& weight_r,  IntList stride, IntList padding, IntList dilation, int groups) 
{  
  auto input = input_r.contiguous();
  auto weight = weight_r.contiguous();
  
  std::vector<int64_t> input_size = {input.size(0), input.size(1), input.size(2), input.size(3)};
  std::vector<int64_t> weight_size = {weight.size(0), weight.size(1), weight.size(2), weight.size(3)};
  auto output_size = conv_output_size(input_size, weight_size, padding, stride, dilation);
  auto output = at::empty(
                    output_size,
                    at::TensorOptions().dtype(torch::kFloat32)
                                       .device(input.device())
                                       .is_variable(true));

  ConvolutionArgs args{ input, output, weight, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT};
  if (false)
  {
    std::clog << "Input size " << input.sizes() << std::endl;
    std::clog << "Kernel size " << weight.sizes() << std::endl;
    std::clog << "Stride  " << stride << std::endl;
    std::clog << "Padding " << padding << std::endl;
    std::clog << "Dilation " << dilation << std::endl;
    std::clog << "Output size " << output.sizes() << std::endl;
  }
  args.handle = at::native::getCudnnHandle();
  args.setConv(padding, stride, dilation, groups, CUDNN_DATA_FLOAT);
  
  cudnnConvolutionFwdAlgo_t fwdAlg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  size_t workspaceSizeInBytes;
  cudnnGetConvolutionForwardWorkspaceSize(args.handle, args.idesc, args.wdesc, args.cdesc.mut_desc(), args.odesc, fwdAlg, &workspaceSizeInBytes);
  auto workspace = at::empty({(signed long)workspaceSizeInBytes}, at::device(output.device()).dtype(torch::kInt8));

  float beta = 0.0f;
  float alpha = 1.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  AT_CUDNN_CHECK(cudnnConvolutionForward(
    args.handle,
    &alpha, args.idesc, input.data_ptr(),
    args.wdesc, weight.data_ptr(),
    args.cdesc.mut_desc(), fwdAlg, workspace.data_ptr(), workspaceSizeInBytes,
    &beta, args.odesc, output.data_ptr()));


  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  
  
  return {output, elapsedTime};
}


std::tuple<Tensor, float> cudnnConv2DBackwardData(
    const Tensor& input_r, const Tensor& weight_r,  IntList stride, IntList padding, IntList dilation, int groups) 
{  
  auto input = input_r.contiguous();
  auto weight = weight_r.contiguous();
  
  std::vector<int64_t> input_size = {input.size(0), input.size(1), input.size(2), input.size(3)};
  std::vector<int64_t> weight_size = {weight.size(0), weight.size(1), weight.size(2), weight.size(3)};

  IntList pad_b = {std::max(0L, 3-1-padding[0]), std::max(0L, 3-1-padding[1])};
  auto output_size = conv_output_size(input_size, weight_size, pad_b, stride, dilation);
  auto output = at::empty(
                    output_size,
                    at::TensorOptions().dtype(torch::kFloat32)
                                       .device(input.device())
                                       .is_variable(false));

  ConvolutionArgs args{ input, output, weight, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT};
  if (true)
  {
    std::clog << "Input size " << input.sizes() << std::endl;
    std::clog << "Kernel size " << weight.sizes() << std::endl;
    std::clog << "Stride  " << stride << std::endl;
    std::clog << "Padding " << pad_b << std::endl;
    std::clog << "Dilation " << dilation << std::endl;
    std::clog << "Output size " << output.sizes() << std::endl;
  }
  args.handle = at::native::getCudnnHandle();
  args.setConv(padding, stride, dilation, groups, CUDNN_DATA_FLOAT);
  
  cudnnConvolutionBwdDataAlgo_t bwdAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  size_t workspaceSizeInBytes;
  cudnnGetConvolutionBackwardDataWorkspaceSize(args.handle, args.wdesc, args.idesc, args.cdesc.mut_desc(), args.odesc, bwdAlgo, &workspaceSizeInBytes);
  auto workspace = at::empty({(signed long)workspaceSizeInBytes}, at::device(output.device()).dtype(torch::kInt8));

  float beta = 0.0f;
  float alpha = 1.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  AT_CUDNN_CHECK(cudnnConvolutionBackwardData(args.handle,
  &alpha, args.wdesc, weight.data_ptr(), 
  args.idesc, input.data_ptr(), 
  args.cdesc.mut_desc(), bwdAlgo, 
  workspace.data_ptr(), workspaceSizeInBytes, &beta, 
  args.odesc, output.data_ptr()));

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  
  
  return {output, elapsedTime};
}


std::tuple<Tensor, float> cudnnConv2DBackwardFilter(
    const Tensor& rinput, const Tensor& rgrad,  IntList stride, IntList padding, IntList dilation, int groups,
    uint32_t kH, uint32_t kW) 
{
  auto input = rinput.contiguous();
  auto ograd = rgrad.contiguous();
  
  IntList output_size = {ograd.size(1), input.size(1), kH, kW};
  auto wgrad = at::empty(
                    output_size,
                    at::TensorOptions().dtype(torch::kFloat32)
                                       .device(input.device())
                                       .is_variable(false));

  ConvolutionArgs args{ input, ograd, wgrad, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT};
  if (false)
  {
    std::clog << "Input size " << input.sizes() << std::endl;
    std::clog << "Kernel size " << ograd.sizes() << std::endl;
    std::clog << "Stride  " << stride << std::endl;
    std::clog << "Padding " << padding << std::endl;
    std::clog << "Dilation " << dilation << std::endl;
    std::clog << "Output size " << wgrad.sizes() << std::endl;
  }
  args.handle = at::native::getCudnnHandle();
  args.setConv(padding, stride, dilation, groups, CUDNN_DATA_FLOAT);
  
  cudnnConvolutionBwdFilterAlgo_t bwdAlgo;
  size_t workspaceSizeInBytes;

  AT_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(args.handle, args.idesc, args.odesc, args.cdesc.mut_desc(), args.wdesc, 
                                                            CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &bwdAlgo));

  cudnnGetConvolutionBackwardFilterWorkspaceSize(args.handle, args.idesc, args.odesc, args.cdesc.mut_desc(), args.wdesc, bwdAlgo, &workspaceSizeInBytes);
  // std::cout << "WORKSPACE " << workspaceSizeInBytes << std::endl;

  auto workspace = at::empty({(signed long)workspaceSizeInBytes}, at::device(wgrad.device()).dtype(torch::kInt8));

  float beta = 0.0f;
  float alpha = 1.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  AT_CUDNN_CHECK(cudnnConvolutionBackwardFilter(args.handle,
  &alpha, args.idesc, input.data_ptr(), 
  args.odesc, ograd.data_ptr(), args.cdesc.mut_desc(),
  bwdAlgo, workspace.data_ptr(), workspaceSizeInBytes, &beta,
  args.wdesc, wgrad.data_ptr()));

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  
  
  return {wgrad, elapsedTime};
}


std::tuple<Tensor, float> cudnnConv2DInt8X32(
    const Tensor& input_r, const Tensor& weight_r,  IntList stride, IntList padding, IntList dilation, int groups) 
{  
  auto input = input_r.contiguous();
  auto weight = weight_r.contiguous();
  
  std::vector<int64_t> input_size = {input.size(0), input.size(1) * 32, input.size(2), input.size(3)};
  std::vector<int64_t> weight_size = {weight.size(0), weight.size(1) * 32, weight.size(2), weight.size(3)};
  
  auto output_size = conv_output_size(input_size, weight_size, padding, stride, dilation);
  output_size[1] /= 32;
  output_size.push_back(32);

  auto output = at::empty(
                    output_size,
                    at::TensorOptions().dtype(torch::kInt8)
                                       .device(input.device())
                                       .layout(input.layout())
                                       .is_variable(true));

  ConvolutionArgs args{ input, output, weight, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x32};
  args.handle = at::native::getCudnnHandle();
  args.setConv(padding, stride, dilation, groups, CUDNN_DATA_INT32);
  
  cudnnConvolutionFwdAlgo_t fwdAlg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  size_t workspaceSizeInBytes;
  cudnnGetConvolutionForwardWorkspaceSize(args.handle, args.idesc, args.wdesc, args.cdesc.mut_desc(), args.odesc, fwdAlg, &workspaceSizeInBytes);
  auto workspace = at::empty({(signed long)workspaceSizeInBytes}, at::device(output.device()).dtype(torch::kInt8));

  float beta = 0.0f;
  float alpha = 1.0f;

  std::clog << "Input size " << input.sizes() << std::endl;
  std::clog << "Kernel size " << weight.sizes() << std::endl;
  std::clog << "Stride  " << stride << std::endl;
  std::clog << "Padding " << padding << std::endl;
  std::clog << "Dilation " << dilation << std::endl;
  std::clog << "Output size " << output.sizes() << std::endl;
  std::clog << "Worspace in Bytes " << workspaceSizeInBytes << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  AT_CUDNN_CHECK(cudnnConvolutionForward(
    args.handle,
    &alpha, args.idesc, input.data_ptr(),
    args.wdesc, weight.data_ptr(),
    args.cdesc.mut_desc(), fwdAlg, workspace.data_ptr(), workspaceSizeInBytes,
    &beta, args.odesc, output.data_ptr()));


  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  
  
  return {output, elapsedTime};
}