from abc import ABC, abstractmethod
from typing import Tuple
import torch.nn as nn
import torch

class BaseEngine(nn.Module, ABC):

    def __init__(self):
        nn.Module.__init__(self)

    @abstractmethod
    def forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

