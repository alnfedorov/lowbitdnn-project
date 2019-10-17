#pragma once

#include <torch/extension.h>

using c10::IntList;
using c10::ArrayRef;
using torch::Tensor;
constexpr uint32_t VECT_C = 16;


Tensor from_vect_c(const Tensor tensor)
{
    auto sizes = tensor.sizes();
    auto result = tensor.permute({0, 1, 4, 2, 3})
                        .reshape({sizes[0], sizes[1]*sizes[4], sizes[2], sizes[3]});
    return result;
}


Tensor to_vect_c(const Tensor tensor, const uint32_t VECT_C=16)
{
    auto shape = tensor.sizes();
    auto result = tensor.reshape({shape[0], shape[1] / VECT_C, VECT_C, shape[2], shape[3]})
                        .permute({0, 1, 3, 4, 2});
    return result;
}
