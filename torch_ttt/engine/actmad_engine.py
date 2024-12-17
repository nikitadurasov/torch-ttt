import torch 
from torchvision.transforms import functional as F
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["ActMADEngine"]

# TODO: add cuda support
@EngineRegistry.register("actmad")
class ActMADEngine(BaseEngine):
    """ **ActMAD** approach: multi-level pixel-wise feature alignment."""
    def __init__(self):
        super().__init__()
        
    def __call__(self, inputs):
        pass