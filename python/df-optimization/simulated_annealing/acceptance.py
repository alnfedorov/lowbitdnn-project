import numpy as np


# General interface is (delta, temp, step, **kwargs) -> float


# min[1, exp(-delta / temp)]
def metropolis_acceptance_prob(delta: float, temp: float, step:int, **kwargs) -> float:
    return min(1, 1 / (1 + np.exp(delta / temp)))


# min{1, [1 - (1-acceptance) * beta * delta] ^ 1/(1 - acceptance)}
def gsa_acceptance_prob(delta: float, temp: float, step: int, q_a:float, **kwargs) -> float:
    temp_step = temp / (step + 1)
    tmp = 1 - (1 - q_a) * delta / temp_step
    if tmp < 0:
        return 0
    else:
        return np.exp(np.log(tmp) / (1 - q_a))
