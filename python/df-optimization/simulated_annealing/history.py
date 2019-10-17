from torch import Tensor
from typing import List, NoReturn


class ParameterHistory:
    """Class used to store information about top-k configurations"""
    def __init__(self, top_k: int = 2):
        self.history = []
        self.top_k = top_k

    def update(self, parameters: List[Tensor], energy: float) -> NoReturn:
        if len(self.history) < self.top_k:
            item = ([x.clone() for x in parameters], energy)
            self.history.append(item)
            return

        for ind, (_, cenegry) in enumerate(self.history):
            if cenegry > energy:
                item = ([x.clone() for x in parameters], energy)
                self.history.insert(ind, item)
                self.history.pop()
                break

        assert len(self.history) == self.top_k

    def best(self) -> List[Tensor]:
        return self.history[0][0]
