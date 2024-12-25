from abc import ABC, abstractmethod
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    @abstractmethod
    def forward(self, model, inputs):
        pass
