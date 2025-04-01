import torch
from typing import Dict, Any, Tuple
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["TentEngine"]

@EngineRegistry.register("tent")
class TentEngine(BaseEngine):
    """**TENT**: Fully test-time adaptation by entropy minimization.

    Args:
        model (torch.nn.Module): The model to adapt.
        optimization_parameters (dict): Hyperparameters for adaptation.

    Reference:
        "TENT: Fully Test-Time Adaptation by Entropy Minimization"
        Dequan Wang, Evan Shelhamer, et al.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimization_parameters: Dict[str, Any] = {},
    ):
        super().__init__()
        self.model = model
        self.optimization_parameters = optimization_parameters

        # Tent adapts only affine parameters in BatchNorm
        self.model.train()
        self._configure_bn()

    def _configure_bn(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.requires_grad_(True)
                module.track_running_stats = False
            else:
                for param in module.parameters(recurse=False):
                    param.requires_grad = False

    def ttt_forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass and entropy loss computation."""
        outputs = self.model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        return outputs, entropy
