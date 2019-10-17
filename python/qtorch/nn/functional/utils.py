import torch
from torch import Tensor


def to_vect_c(tensor: Tensor, contiguous=False):
    N, C, H, W = tensor.shape

    if C % 4 != 0:
        raise NotImplementedError("VECT_C works only with tensors which channels are either multiple of 4")

    # vect_c = 32 if C % 32 == 0 else 4
    vect_c = 4

    tensor = tensor.reshape(N, C // vect_c, vect_c, H, W) \
                   .permute([0, 1, 3, 4, 2])
    if contiguous:
        tensor = tensor.contiguous()
    return tensor


def from_vect_c(tensor: Tensor, contiguous=True):
    N, C, H, W, VECT_C = tensor.shape

    tensor = tensor.permute([0, 1, 4, 2, 3]) \
                   .reshape(N, C*VECT_C, H, W)

    if contiguous:
        tensor = tensor.contiguous()
    return tensor
