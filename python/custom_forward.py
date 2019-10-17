import pytest

# pytest.main()

import torch
import time
import numpy as np

from qtorch.jit import cpp
from torch.nn import Conv2d
from tqdm import tqdm
# from qtorch.nn.functional import quantize, to_vect_c, dequantize, from_vect_c, QUANTIZATION_PARAMETERS
torch.manual_seed(23)

REPEATS = 100
IN_CHANNELS = 32
OUT_CHANNELS = 32

# vary
BACTH = 1
KH, KW = 3, 3
H, W = 256, 256
KH, KW = 3, 3

PADDING, STRIDE, DILATION, GROUPS = [0, 0], [1, 1], [1, 1], 1


def to_vect_c(tensor):
    N, C, H, W = tensor.shape
    assert C // 32 * 32 == C
    vect_c = 32
    tensor = tensor.reshape(N, C // vect_c, vect_c, H, W) \
                   .permute([0, 1, 3, 4, 2]).contiguous()
    return tensor

class conv(torch.nn.Module):
    def forward(self, data, kernel):
        return cpp.conv2d_forwardv1(data, kernel)

conv = conv()

# check correctness
fconv = Conv2d(IN_CHANNELS, OUT_CHANNELS, (KH, KW), STRIDE, PADDING, DILATION, GROUPS, bias=False).to('cuda')
fconv.weight.data.uniform_(0, 2).round_()

with torch.no_grad():
    executed_cudnn = []
    executed_int8 = []

    kernel = to_vect_c(fconv.weight.data.to(torch.int8)).contiguous()
    data = torch.rand((BACTH, IN_CHANNELS, H, W), dtype=torch.float32, device="cuda", requires_grad=False) \
        .mul_(5).sub_(1).to(torch.int8)
    my_data = to_vect_c(data).contiguous()

    for i in tqdm(range(REPEATS)):

        # cudnn
        cudnn_data = data.float()
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            fwd_cudnn = fconv(cudnn_data)
        timing = [x.cuda_time for x in prof.key_averages() if "cudnn" in x.key]
        assert len(timing) == 1
        executed_cudnn.append(timing[0] / 1000) # milliseconds

        # int8
        fwd_my, timing = conv(my_data, kernel)
        executed_int8.append(timing)  # milliseconds

        assert (fwd_my.float() - to_vect_c(fwd_cudnn).float()).abs().sum() == 0

    executed_cudnn = np.asarray(executed_cudnn)
    print("cudnn forward", executed_cudnn.mean(), executed_cudnn.std())

    executed_int8 = np.asarray(executed_int8)
    print("my forward", executed_int8.mean(), executed_int8.std())




# fconv = Conv2d(IN_CHANNELS, OUT_CHANNELS, (KH, KW), STRIDE, PADDING, DILATION, GROUPS).to('cuda')
# with torch.no_grad():
#     executed = []
#     for i in range(10000):
#         begin = time.time()
#         bwd_cudnn = fconv(t)
#         end = time.time()
#         executed.append(end - begin)
#     executed = np.asarray(executed)
#     print("cudnn forward ", executed.mean(), executed.std())
#
#
# qconv2d = QConv2D(IN_CHANNELS, OUT_CHANNELS, (KH, KW), STRIDE, PADDING, DILATION, GROUPS).to('cuda')
# weights = qconv2d.weight.to('cuda')
#
#
# with torch.no_grad():
#     executed = []
#     assert weights.device == t.device
#     for i in range(10000):
#         begin = time.time()
#         input = quantize(t, to_vect_c=True, stochastic=False)
#         weight = quantize(weights, to_vect_c=False, stochastic=False)
#         end = time.time()
#         QUANTIZATION_PARAMETERS.clear()
#         executed.append(end - begin)
#     executed = np.asarray(executed)
#     print("memory allocation and copy ", executed.mean(), executed.std())
#
# input = quantize(t, to_vect_c=True, stochastic=False)
# weight = quantize(weights, to_vect_c=False, stochastic=False)
# iscale, wscale = QUANTIZATION_PARAMETERS[input], QUANTIZATION_PARAMETERS[weight]
# qconv_scale = wscale * iscale
#
# with torch.no_grad():
#     executed = []
#     for i in range(10000):
#         begin = time.time()
#         result = cpp.conv2d(input, weight, "external", qconv_scale, STRIDE, PADDING, DILATION, GROUPS)
#         end = time.time()
#         executed.append(end - begin)
#     executed = np.asarray(executed)
#     print("conv2d int8 ", executed.mean(), executed.std())