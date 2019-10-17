import numpy as np


# General interface is (initial_temp, step, **kwargs) -> float


def linear_schedule(initial_temp: float, step: int, **kwargs) -> float:
    assert step > 0 and initial_temp > 0
    return initial_temp / step


def log_schedule(initial_temp: float, step: int, **kwargs) -> float:
    assert step > 0 and initial_temp > 0
    return initial_temp / np.log(step + 1)


def gsa_schedule(initial_temp: float, step: int, q_v:float, **kwargs):
    assert step > 0 and initial_temp > 0
    t1 = np.exp((q_v - 1.) * np.log(2.)) - 1.
    t2 = np.exp((q_v - 1.) * np.log(step + 1)) - 1.
    return initial_temp * t1 / t2
