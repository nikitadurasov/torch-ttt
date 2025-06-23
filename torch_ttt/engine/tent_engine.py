import torch
from typing import Dict, Any, Tuple
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["TentEngine"]

@EngineRegistry.register("tent")
class TentEngine(BaseEngine):
    """**TENT**: Fully test-time adaptation by entropy minimization.

    TENT adapts models at inference by minimizing prediction entropy, encouraging confident outputs on unlabeled data. It updates only BatchNorm affine parameters and requires no labels or training supervision.

    Args:
        model (torch.nn.Module): Model to be adapted at test-time.
        optimization_parameters (dict): Optimizer configuration for adaptation (e.g. learning rate).

    :Example:

    .. code-block:: python

        from torch_ttt.engine.tent_engine import TentEngine

        model = MyModel()
        engine = TentEngine(model, {"lr": 1e-3})
        optimizer = torch.optim.Adam(engine.parameters(), lr=1e-3)

        # Training
        engine.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs, loss_ttt = engine(inputs)
            loss = criterion(outputs, labels) + alpha * loss_ttt
            loss.backward()
            optimizer.step()

        # Inference
        engine.eval()
        for inputs, labels in test_loader:
            output, loss_ttt = engine(inputs)

    Reference:

        "Tent: Fully Test-Time Adaptation by Entropy Minimization",  
        Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Vasconcelos, Trevor Darrell

        Paper link: `PDF <https://arxiv.org/pdf/2006.10726.pdf>`_
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
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            The current model prediction and the entropy loss value.
        """
        outputs = self.model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        return outputs, entropy
