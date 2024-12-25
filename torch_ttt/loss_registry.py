import importlib
import os
from torch_ttt.loss.base_loss import BaseLoss


class LossRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(loss_class):
            if not issubclass(loss_class, BaseLoss):
                raise TypeError(f"Loss class '{name}' must inherit from BaseLoss.")
            if name in cls._registry:
                raise ValueError(f"Loss '{name}' is already registered.")

            cls._registry[name] = loss_class
            return loss_class

        return decorator

    @classmethod
    def get_loss(cls, name):
        if name not in cls._registry:
            raise ValueError(f"Loss '{name}' is not registered.")
        return cls._registry[name]


# Dynamically import all losses in the losses directory
def register_all_losses():
    losses_dir = os.path.dirname(__file__) + "/loss"
    for file in os.listdir(losses_dir):
        if file.endswith("_loss.py") and not file.startswith("__"):
            module_name = f"torch_ttt.loss.{file[:-3]}"
            importlib.import_module(module_name)


register_all_losses()
