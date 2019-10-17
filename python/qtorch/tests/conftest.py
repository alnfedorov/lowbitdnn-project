import torch
import pytest
from qtorch.nn import QConv2D
from torch.nn import Conv2d

torch.manual_seed(32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# function scope to avoid gpu out of memory errors
@pytest.fixture(scope='function', params=[
    (batch, channels, H, W)
    for batch in [1, 3] for channels in [12, 64, 512]
    for H, W in [(1, 1), (1, 5), (6, 3), (64, 112), (224, 224)]
])
def random_tensor(request):
    shape = request.param
    return torch.rand(shape, dtype=torch.float32, device=device, requires_grad=False).sub_(0.5).mul_(2)


@pytest.fixture(params=[(1, 1), (3, 3)])
def kernel(request):
    return request.param


@pytest.fixture(params=[(1, 1), 2])
def stride(request):
    return request.param


@pytest.fixture(params=[(1, 2), (3, 3), 0])
def padding(request):
    return request.param


_dilation_fail = lambda *x: pytest.param(x, marks=pytest.mark.xfail(reason="Dilation should be = 1"))
@pytest.fixture(params=[
    (1, 1),
    # _dilation_fail(3, 2), _dilation_fail(0, -2)
])
def dilation(request):
    return request.param


# @pytest.fixture(params=[
#     1, 4, 32,
#     pytest.param(0, marks=pytest.mark.xfail(reason="Groups should be > 0"))
# ])
# def groups(request):
#     return request.param


CONV_PARAMS = (4, 20, 32, 128)

@pytest.fixture(params=CONV_PARAMS)
def conv2d(kernel, stride, padding, dilation, request):
    out_channels = request.param

    if dilation[0] != 1 or dilation[1] != 1: # or out_channels < groups:
        pytest.skip()

    qconv2d = lambda in_channels: \
        QConv2D(in_channels, out_channels, kernel, stride, padding, dilation, 1, stochastic=False, min_value=-3, max_value=6).to(device)

    fconv2d = lambda in_channels: \
        Conv2d(in_channels, out_channels, kernel, stride, padding, dilation, 1, bias=False).to(device)

    return fconv2d, qconv2d
