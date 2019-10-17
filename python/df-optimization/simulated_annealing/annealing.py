import torch
from torch import Tensor
from functools import lru_cache
import numpy as np
from scipy.special import gammaln
from typing import List, NoReturn


# The following procedures are used to in-place generate new parameters
# General interface is (parameters: List[Tensor], T: float, **kwargs) -> void


def fast_annealing(parameters: List[Tensor], temp: float, lower=-25, upper=25, **kwargs) -> NoReturn:
    for x in parameters:
        alpha = torch.rand_like(x, dtype=torch.float32)
        z = torch.sign(alpha - 0.5) * temp * (torch.pow((1 + 1 / temp), torch.abs(2 * alpha - 1)) - 1)
        # mul = (torch.rand_like(z) >= 0.5).to(z)
        # z.mul_(mul)
        z.mul_(upper - lower).add_(x.to(z)).round_().clamp_(lower, upper)
        torch.s_copy_(x, z.to(x))
        # assert x.min() >= lower and x.max() <= upper


@lru_cache(maxsize=None)
def _visita(q_v:float, temp: float) -> float:
    factor1 = np.exp(np.log(temp) / (q_v - 1.0))
    factor2 = np.exp((4.0 - q_v) * np.log(q_v - 1.0))
    factor3 = np.exp((2.0 - q_v) * np.log(2.0) / (q_v - 1.0))
    factor4 = np.sqrt(np.pi) * factor1 * factor2 / (factor3 * (3.0 - q_v))
    factor5 = 1.0 / (q_v - 1.0) - 0.5
    d1 = 2.0 - factor5
    factor6 = np.pi * (1.0 - factor5) / \
              np.sin(np.pi * (1.0 - factor5)) / np.exp(gammaln(d1))
    sigmax = np.exp(-(q_v - 1.0) * np.log(factor6 / factor4) / (3.0 - q_v))
    x = sigmax * np.random.normal()
    y = np.random.normal()
    den = np.exp(
        (q_v - 1.0) * np.log((np.fabs(y))) / (3.0 - q_v)
    )
    return x / den


def generalized_annealing(parameters: List[Tensor], temp: float, q_v: float, lower, upper, **kwargs) -> NoReturn:
    visit = _visita(q_v, temp)
    if visit < -1e-8:
        visit = -1e-8 * np.random.random()
    elif visit > 1e8:
        visit = 1e8 * np.random.random()

    bound_range = upper - lower
    visit = int(round(visit % bound_range))                         # TODO Not valid for float optimizations
    assert isinstance(visit, int) and isinstance(bound_range, int)

    for x in parameters:
        x_ = x.to(torch.int16)
        x_.add_(visit - lower).fmod_(bound_range).add_(bound_range).fmod_(bound_range).add_(lower)
        # assert x_.min() >= lower and x_.max() <= upper # TODO Remove, it's rather time consuming
        torch.s_copy_(x, x_.to(x))
