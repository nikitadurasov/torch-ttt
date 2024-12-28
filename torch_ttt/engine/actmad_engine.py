import torch
from typing import List, Dict, Any, Tuple
from contextlib import contextmanager
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["ActMADEngine"]


# TODO: add cuda support
@EngineRegistry.register("actmad")
class ActMADEngine(BaseEngine):
    """**ActMAD** approach: multi-level pixel-wise feature alignment.
    
    Args:
        model (torch.nn.Module): Model to be trained with TTT.
        features_layer_names (List[str]): The names of the layers from which the features are extracted.
        optimization_parameters (dict): The optimization parameters for the engine.

    """

    def __init__(
            self,
            model: torch.nn.Module,
            features_layer_names: List[str],
            optimization_parameters: Dict[str, Any] = {},
    ):
        super().__init__()
        self.model = model
        self.features_layer_names = features_layer_names
        self.optimization_parameters = optimization_parameters

        # TODO: rewrite this
        self.target_modules = []
        for layer_name in self.features_layer_names:
            layer_exists = False
            for name, module in model.named_modules():
                if name == layer_name:
                    layer_exists = True
                    self.target_modules.append(module)
                    break
            if not layer_exists:
                raise ValueError(f"Layer {layer_name} does not exist in the model.")

    def ttt_forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            Returns the current model prediction and rotation loss value.
        """
        raise NotImplementedError

    @contextmanager
    def __capture_hook(self):
        """Context manager to capture features via a forward hook."""
    
        class OutputHook:
            def __init__(self):
                self.output = None

            def hook(self, module, input, output):
                self.output = output

        features_hooks = []
        for module in self.target_modules:
            hook = OutputHook()
            features_hooks.append(hook)
            module.register_forward_hook(hook.hook)

        try:
            yield features_hooks
        finally:
            for hook in features_hooks:
                hook.remove()