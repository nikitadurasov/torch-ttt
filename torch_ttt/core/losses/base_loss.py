from abc import ABC, abstractmethod

class BaseLoss(ABC):

    @abstractmethod
    def __call__(self, model, inputs):
        pass

