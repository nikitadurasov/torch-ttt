from abc import ABC, abstractmethod
import torch.nn as nn

class BaseEngine(nn.Module, ABC):

    def __init__(self):
        nn.Module.__init__(self)

    @abstractmethod
    def __call__(self, inputs):
        pass

