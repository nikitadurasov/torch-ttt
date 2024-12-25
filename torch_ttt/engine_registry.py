import importlib
import os
from torch_ttt.engine.base_engine import BaseEngine


class EngineRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(engine_class):
            if not issubclass(engine_class, BaseEngine):
                raise TypeError(f"Engine class '{name}' must inherit from BaseEngine.")
            if name in cls._registry:
                raise ValueError(f"Engine '{name}' is already registered.")

            cls._registry[name] = engine_class
            return engine_class

        return decorator

    @classmethod
    def get_engine(cls, name):
        if name not in cls._registry:
            raise ValueError(f"Engine '{name}' is not registered.")
        return cls._registry[name]


# Dynamically import all losses in the losses directory
def register_all_engines():
    losses_dir = os.path.dirname(__file__) + "/engine"
    for file in os.listdir(losses_dir):
        if file.endswith("_engine.py") and not file.startswith("__"):
            module_name = f"torch_ttt.engine.{file[:-3]}"
            importlib.import_module(module_name)


register_all_engines()
