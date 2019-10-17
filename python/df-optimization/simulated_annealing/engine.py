import gc
import torch
from random import random
from torch import Tensor
from typing import NoReturn, List, Callable
from .history import ParameterHistory


def simulated_annealing(parameters: List[Tensor], fun: Callable,
                        annealing_strategy: Callable, temperature_schedule: Callable, acceptance_strategy: Callable,
                        initial_temp: float=5230, restart_temp_ratio: float=2e-5, max_no_success_attempts: int=100,
                        max_eval: int=10000) -> ParameterHistory:
    history = ParameterHistory(top_k=3)
    assert 0 < restart_temp_ratio < 1, 'Restart temperature ratio has to be in range (0, 1)'

    temp_restart = initial_temp * restart_temp_ratio
    finished = False
    energy, eval, step = float('inf'), 0, 1

    while not finished:
        temp = temperature_schedule(initial_temp, step)
        if temp <= temp_restart:
            step = 1
            continue
        print(f"Accuracy is {-energy/10 + 1}, temperature is {temp}")
        backup = [x.clone() for x in parameters]

        # Try to find better configuration
        # attempt = 1
        # while attempt < max_no_success_attempts and not finished:
        for x in [*[[x] for x in parameters], parameters]:
            annealing_strategy(x, temp)
            new_energy = fun()

            eval += 1
            if eval >= max_eval:
                finished = True

            delta = new_energy - energy
            accept_probability = acceptance_strategy(delta, temp, step)

            if random() <= accept_probability:
                history.update(parameters, new_energy)
                energy = new_energy
                break

            # Restore previous state
            # attempt += 1
            for x, x_old in zip(parameters, backup):
                torch.s_copy_(x, x_old)

        # Restart if everything is bad
        # if attempt >= max_no_success_attempts:
        #     step = 1
        #     continue
        step += 1
        # TODO: some sort of the local search?
    return history
