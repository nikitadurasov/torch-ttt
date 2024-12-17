import torch 
from torchvision.transforms import functional as F
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["TTTPPEngine"]

# TODO: add cuda support
@EngineRegistry.register("ttt_pp")
class TTTPPEngine(BaseEngine):
    """ **TTT++** approach: feature alignment-based + SimCLR loss.

    Args:
        model (torch.nn.Module): Model to be trained with TTT.
        features_layer_name (str): The name of the layer from which the features are extracted.
    
    Reference:
        "TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?", Yuejiang Liu, Parth Kothari, Bastien van Delft, Baptiste Bellot-Gurlet, Taylor Mordan, Alexandre Alahi

        Paper link: https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html
    
    """
    def __init__(
            self,
            model: torch.nn.Module,
            features_layer_name: str, 
    ) -> None:
        super().__init__()
        self.model = model
        self.features_layer_name = features_layer_name
        
    def __call__(self, inputs):
        pass