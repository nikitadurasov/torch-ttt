from abc import ABC, abstractmethod

class BaseEngine(ABC):

    @abstractmethod
    def __call__(self, inputs):
        pass

