import torch
import numpy as np
from torch import jit, Tensor
from typing import Tuple, Any
from torch.autograd.function import Function
from .utils import to_vect_c, from_vect_c

# Note:
# ----------------------------------------------------------------------------------------------------------
# rscale * int8_conv_result = iscale * wscale * CONVOLUTION(input, weight)  -> float values
# int8_conv_result = iscale * wscale / rscale * CONVOLUTION(input, weight)  -> qtorch values
#
# Here iscale is input scale, wscale is weights scale, rscale is result scale
# Expression  iscale * wscale / rscale will be referenced as qconv_scale in the code
# ----------------------------------------------------------------------------------------------------------

__all__ = ['quantize', 'dequantize', "QUANTIZATION_PARAMETERS"]

# Only int8 is supported by cudnn with proper speed up
NUM_BITS = 8
DTYPE = torch.int8

# Work around to store quantization parameters
QUANTIZATION_PARAMETERS = {}


class _Quantize(Function):
    @staticmethod
    def forward(ctx: Any, tensor: Tensor, _to_vect_c: bool, qmin: int, qmax: int, qzero: int, scale: float,
                inv_scale: float, stochastic: bool, dtype: Any):
        if _to_vect_c:
            tensor = to_vect_c(tensor)
        else:
            tensor = tensor.clone()
        # ctx.converted_to_vect_c = _to_vect_c
        tensor.mul_(inv_scale)  # to fake integers

        if qzero != 0:
            tensor.add_(qzero)

        if stochastic:  # Random rounding to improve robustness
            noise = tensor.new_empty(tensor.shape).uniform_(-0.5, 0.5)
            tensor.add_(noise)

        # Stash mask with variables outside the range for back propagation
        # ctx.save_for_backward((tensor <= qmin) | (tensor >= qmax))

        tensor.clamp_(qmin, qmax).round_()  # clamp fake integers to qmin, qmax
        return tensor.to(dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # import pydevd
        # pydevd.settrace(suspend=True, trace_only_current_thread=True)

        raise NotImplementedError()
        # scale = QUANTIZATION_PARAMETERS[grad_output]
        #
        # grad_input = grad_output.clone()
        #
        # out_of_range = ctx.saved_tensors
        # grad_input[out_of_range] = 0
        #
        # from_vect_c = ctx.converted_to_vect_c
        # grad_input = dequantize(grad_input, from_vect_c=from_vect_c, scale=scale)

        # return grad_input, None, None, None, None, None, None, None, None


class _Dequantize(Function):
    @staticmethod
    def forward(ctx: Any, tensor: Tensor, scale: float, _from_vec_c: bool):
        if _from_vec_c:
            tensor = from_vect_c(tensor)

        ctx.converted_from_vect_c = _from_vec_c
        return tensor.float() * scale

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # import pydevd
        # pydevd.settrace(suspend=True, trace_only_current_thread=True)

        raise NotImplementedError()
        # to_vect_c = ctx.converted_from_vect_c
        # grad_output = quantize(grad_output, to_vect_c)
        # return grad_output, None, None


def _quantization_params(num_bits: int, min_value: float,
                         max_value: float, signed: bool) -> Tuple[int, int, int, float, float]:
    if signed:
        qmin, qmax = - 2**num_bits // 2, 2**num_bits // 2 - 1
    else:
        qmin, qmax = 0, 2**num_bits - 1
    assert isinstance(qmin, int) and isinstance(qmax, int)

    # Determine the scale
    scale = (max_value - min_value) / (qmax - qmin)     # qvalue * scale = float_value
    inv_scale = (qmax - qmin) / (max_value - min_value) # float_value * inv_scale = qvalue

    # Make zero exactly representable. Zero point is obtained from the following equation:
    # (qmin - qzero) * scale = min_value
    # qzero = qmin - min_value * inv_scale
    # Actually it slightly shifts requested range, but it's inevitable evil in order to exactly represent 0
    qzero = qmin - min_value * inv_scale
    qzero = int(round(qzero))
    qzero = np.clip(qzero, qmin, qmax)  # type: int # boundary situations
    # Left scale and inv_scale as float32 because weights and bias are float32 by default
    return qmin, qmax, qzero, scale, inv_scale


def quantize(tensor, to_vect_c, num_bits=None, min_value=None, max_value=None, stochastic=False):
    # type: (Tensor, bool, int, float, float, bool) -> Tensor

    if tensor in QUANTIZATION_PARAMETERS:
        return tensor

    num_bits = NUM_BITS if not num_bits else num_bits
    assert num_bits <= NUM_BITS, "num bits > 8 are not supported"

    with torch.no_grad():
        min_value = tensor.min() if not min_value else min_value
        max_value = tensor.max() if not max_value else max_value

        min_value, max_value = float(min_value), float(max_value)

        # Force symmetric ranges. RECHECK FOR ONLY POSITIVE AND ONLY NEGATIVE VALUES
        max_abs = max(abs(min_value), max_value)
        min_value, max_value = -max_abs, max_abs

        # num_bits = max(1, num_bits - 1)
        qmin, qmax, qzero, scale, inv_scale = _quantization_params(num_bits, min_value, max_value, signed=True)
        qzero = 0
        assert qzero == 0, "not symmetrical quantization ranges are not supproted"
    result = _Quantize().apply(tensor, to_vect_c, qmin, qmax, qzero, scale, inv_scale, stochastic, DTYPE)

    QUANTIZATION_PARAMETERS[result] = scale
    # result.quantized = True
    # result.scale = scale
    return result


def dequantize(tensor: Tensor, from_vect_c: bool, scale: float = None) -> Tensor:
    if not scale:
        assert tensor in QUANTIZATION_PARAMETERS, "Tried to dequantize not quantized tensor"
        scale = QUANTIZATION_PARAMETERS[tensor]

    assert tensor.dtype == torch.int8

    tensor = _Dequantize().apply(tensor, scale, from_vect_c)
    return tensor
    # qzero = tensor.qzero
    # if qzero != 0:
    #     tensor.sub_(qzero)

    # return tensor.mul_(scale)
