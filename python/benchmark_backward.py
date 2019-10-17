import pytest

# pytest.main()

import torch
import time
from qtorch.nn.functional import quantize, to_vect_c, dequantize, from_vect_c, QUANTIZATION_PARAMETERS
import numpy as np
from qtorch.jit import cpp
from qtorch.nn import QConv2D
from torch.nn import Conv2d
torch.manual_seed(23)


# Naive forward

BACTH, H, W, IN_CHANNELS = 64, 128, 128, 256
OUT_CHANNELS, KH, KW = 256, 3, 3
PADDING, STRIDE, DILATION, GROUPS = [1, 1], [1, 1], [1, 1], 1

t = torch.rand((BACTH, IN_CHANNELS, H, W), dtype=torch.float32, device="cuda", requires_grad=False).sub_(0.5).mul_(2)
g = torch.rand((BACTH, OUT_CHANNELS, H, W), dtype=torch.float32, device="cuda", requires_grad=False).sub_(0.5).mul_(2)
# g = torch.ones((BACTH, OUT_CHANNELS, H, W), dtype=torch.float32, device="cuda", requires_grad=False)


def tmp(t, g, strides, padding, dilation, groups):
    input = t
    grad_output = g


    # Input gradients. Simply
    input_view = input.permute([1, 0, 2, 3])  # C, B, H, W
    input_view = quantize(input_view, to_vect_c=True)
    iscale = QUANTIZATION_PARAMETERS[input_view]



    grad_view = quantize(grad_output.permute([1, 0, 2, 3]), to_vect_c=True)
    gscale = QUANTIZATION_PARAMETERS[grad_view]

    assert input_view.shape[-1] == grad_view.shape[-1]

    qconv_scale = iscale * gscale
    grad_weight = cpp.conv2d(input_view, grad_view, "external", qconv_scale, strides, padding, dilation, groups)
    grad_weight = grad_weight.permute([1, 0, 2, 3])
    return grad_weight


with torch.no_grad():
    executed = []
    for i in range(10000):
        begin = time.time()
        bwd_cudnnint8 = tmp(t, g, STRIDE, PADDING, DILATION, GROUPS)
        end = time.time()
        executed.append(end-begin)
        QUANTIZATION_PARAMETERS.clear()
    executed = np.asarray(executed)
    print("cudnn int8 backward for data ", executed.mean(), executed.std())


with torch.no_grad():
    executed = []
    for i in range(10000):
        begin = time.time()
        bwd_cudnn = cpp.conv2d_backward_data(t, g, STRIDE, PADDING, DILATION, GROUPS)
        end = time.time()
        executed.append(end-begin)
    executed = np.asarray(executed)
    print("cudnn backward for data ", executed.mean(), executed.std())


fconv = Conv2d(IN_CHANNELS, OUT_CHANNELS, (KH, KW), STRIDE, PADDING, DILATION, GROUPS).to('cuda')


with torch.no_grad():
    executed = []
    for i in range(10000):
        begin = time.time()
        bwd_cudnn = fconv(t)
        end = time.time()
        executed.append(end - begin)
    executed = np.asarray(executed)
    print("cudnn forward ", executed.mean(), executed.std())


qconv2d = QConv2D(IN_CHANNELS, OUT_CHANNELS, (KH, KW), STRIDE, PADDING, DILATION, GROUPS).to('cuda')
weights = qconv2d.weight.to('cuda')


with torch.no_grad():
    executed = []
    assert weights.device == t.device
    for i in range(10000):
        begin = time.time()
        input = quantize(t, to_vect_c=True, stochastic=False)
        weight = quantize(weights, to_vect_c=False, stochastic=False)
        end = time.time()
        QUANTIZATION_PARAMETERS.clear()
        executed.append(end - begin)
    executed = np.asarray(executed)
    print("memory allocation and copy ", executed.mean(), executed.std())

input = quantize(t, to_vect_c=True, stochastic=False)
weight = quantize(weights, to_vect_c=False, stochastic=False)
iscale, wscale = QUANTIZATION_PARAMETERS[input], QUANTIZATION_PARAMETERS[weight]
qconv_scale = wscale * iscale

with torch.no_grad():
    executed = []
    for i in range(10000):
        begin = time.time()
        result = cpp.conv2d(input, weight, "external", qconv_scale, STRIDE, PADDING, DILATION, GROUPS)
        end = time.time()
        executed.append(end - begin)
    executed = np.asarray(executed)
    print("conv2d int8 ", executed.mean(), executed.std())