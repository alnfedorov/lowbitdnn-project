import torch
import numpy as np
from qtorch.jit import cpp
from qtorch.nn.functional import quantize, to_vect_c, from_vect_c, QUANTIZATION_PARAMETERS
from torch import Tensor, jit
from torch.autograd import Function
from typing import Tuple, Union

# cpp.conv2d(input, weight, scale, stride, padding, dilation, groups)
# cpp.max_pool2d(input, kernel, stride, padding)

__all__ = ["qconv2d"]

_IntOptTuple = Union[Tuple[int, int], int]

# def _compute_result_scale(kernel: Tuple[int, int], channels: int, iscale: float, wscale: float) -> Tuple[float, float]:
#     if abs(iscale - wscale) < 1e-8:
#         rscale = iscale # set scale_result to iscale for simplicity
#     else:
#         rscale = (iscale * wscale)**0.5
#         # rscale = max(iscale, wscale)
#
#     # imax, wmax = 255 * iscale, 255 * wscale
#     # nrscale = imax * wmax * kernel[0] * kernel[1] * channels / 255
#     # rscale = max(rscale / 10, nrscale)
#
#     # Assuming that input(x) and weights(k) are uniformly distributed values, one can write:
#     # result(x, y) = sum k_c_i * x_c_i for i = 0 to kernel size for c = 0 to channels
#     # E(result(x, y)) = sum E(k_c_i * x_c_i) = kernel_size * channels * E(k * x)
#
#     # https://www.wolframalpha.com/input/?i=integral+z%2Fx+*+dx+for+x%3Dz+to+128
#     # https://math.stackexchange.com/questions/659254/product-distribution-of-two-uniform-distribution-what-about-3-or-more
#     # f(k * x) = log(128/z)
#
#     # https://www.wolframalpha.com/input/?i=integral+log(128%2Fx)*x*dx+for+x%3D0++to+128
#     # Assuming 8 bit values, we got
#     # E(k * x) = 4096
#
#     # E(result(x, y)) = 4096 * kernel_h * kernel_w * channels
#
#
#     # mean_abs_int8 = 4096 * kernel[0] * kernel[1] * channels
#
#     qconv_scale = iscale * wscale / rscale
#     return rscale, qconv_scale


class _QConv2d(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor,
                strides: Tuple[int, int], padding: Tuple[int, int], dilation: Tuple[int, int], groups: int) -> Tensor:
        ctx.original_input_shape = input.shape
        ctx.original_weight_shape = weight.shape

        input = quantize(input, to_vect_c=True, stochastic=False)
        weight = quantize(weight, to_vect_c=False, stochastic=False)
        iscale, wscale = QUANTIZATION_PARAMETERS[input], QUANTIZATION_PARAMETERS[weight]

        ctx.iscale, ctx.wscale = iscale, wscale
        ctx.strides = strides
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        ctx.save_for_backward(input, weight)

        qconv_scale = wscale * iscale

        result = cpp.conv2d(input, weight, "external", qconv_scale, strides, padding, dilation, groups)
        assert result.dtype == torch.float32
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        iscale, wscale, strides, padding, dilation, groups = ctx.iscale, ctx.wscale, ctx.strides, \
                                                             ctx.padding, ctx.dilation, ctx.groups
        # import pydevd
        # pydevd.settrace(suspend=True, trace_only_current_thread=True)

        grad_output = quantize(grad_output, to_vect_c=True)
        gscale = QUANTIZATION_PARAMETERS[grad_output]

        assert dilation == (1, 1), "Only (1, 1) dilation is supported"
        assert groups == 1, "Only 1 groups is supported"
        assert padding[0] % weight.shape[-2] == 0 and padding[1] % weight.shape[-1] == 0, \
            "Padding larger then kernel size doesn't make sense"
        assert strides == (1, 1), "No way to compute for strides != 1. Only by inserting dummy rows and columns"

        # Input gradients. Simply
        input_view = from_vect_c(input, contiguous=False).permute([1, 0, 2, 3])  # C, B, H, W
        input_view = to_vect_c(input_view)

        grad_view = from_vect_c(grad_output, contiguous=False).permute([1, 0, 2, 3])  # C, B, H, W
        grad_view = to_vect_c(grad_view)

        assert input_view.shape[-1] == grad_view.shape[-1]

        qconv_scale = iscale * gscale
        grad_weight = cpp.conv2d(input_view, grad_view, "external", qconv_scale, strides, padding, dilation, groups)
        grad_weight = to_vect_c(grad_weight.permute([1, 0, 2, 3]), contiguous=True)
        # grad_weight = quantize(grad_weight.permute([1, 0, 2, 3]), to_vect_c=True)
        assert grad_weight.shape == ctx.original_weight_shape

        weight_view = from_vect_c(weight, contiguous=False)
        weight_view = torch.rot90(weight_view, k=-2, dims=(2, 3))
        weight_view = to_vect_c(weight_view.permute([1, 0, 2, 3]))
        padding = [max(0, x-1-p) for x, p in zip(weight.shape[2:4], padding)]
        assert all(x >= 0 for x in padding)

        qconv_scale = gscale * wscale
        grad_input = cpp.conv2d(grad_output, weight_view, "external", qconv_scale, strides, padding, dilation, groups)
        # grad_input = quantize(grad_input, to_vect_c=True)
        assert grad_input.shape == ctx.original_input_shape

        return grad_input, grad_weight, None, None, None, None


def qconv2d(input, weight, stride, padding, dilation, groups):
    # type: # (Tensor, float, Tensor, float, _IntOptTuple, _IntOptTuple, _IntOptTuple, int) -> Tensor
    # assert config == "external", "native config is not supported"
    result = _QConv2d().apply(input, weight, stride, padding, dilation, groups)
    return result
