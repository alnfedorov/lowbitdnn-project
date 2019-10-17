#pragma once

#include <cudnn_v7.h>
#include <torch/extension.h>
#include <ATen/ATen.h>


using at::Tensor;
cudnnDataType_t getCudnnDataType(const Tensor& tensor)
{
  AT_CHECK(tensor.dim() == 5);
  if (tensor.size(4) == 32)
    return CUDNN_DATA_INT8x32;
  if (tensor.size(4) == 4)
    return CUDNN_DATA_INT8x4;
  else
    throw std::runtime_error("Tensor must be exactly divisible by 4 or 32");
}


cudnnTensorFormat_t getCudnnTensorFormat(const Tensor& tensor)
{
  AT_CHECK(tensor.dim() == 5);
  if (tensor.size(4) == 32 || tensor.size(4) == 4)
    return CUDNN_TENSOR_NCHW_VECT_C;
  else
      throw std::runtime_error("Tensor must be exactly divisible by 4 or 32");
}
