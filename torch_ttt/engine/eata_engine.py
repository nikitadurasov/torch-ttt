import torch
from typing import Dict, Any, Tuple
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["EATAEngine"]

@EngineRegistry.register("eata")
class EATAEngine(BaseEngine):

    def __init__(
        self,
        model: torch.nn.Module,
        optimization_parameters: Dict[str, Any] = {},
    ):
        super().__init__()
        self.model = model
        self.optimization_parameters = optimization_parameters

    def ttt_forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass