import torch
from typing import List, Dict, Any, Tuple
from contextlib import contextmanager

from torch.utils.data import DataLoader
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

    :Example:

    .. code-block:: python

        from torch_ttt.engine.actmad_engine import ActMADEngine

        model = MyModel()
        engine = ActMADEngine(model, ["fc1", "fc2"])
        optimizer = torch.optim.Adam(engine.parameters(), lr=1e-4)

        # Training
        engine.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs, loss_ttt = engine(inputs)
            loss = criterion(outputs, labels) + alpha * loss_ttt
            loss.backward()
            optimizer.step()

        # Compute statistics for features alignment
        engine.compute_statistics(train_loader)

        # Inference
        engine.eval()
        for inputs, labels in test_loader:
            output, loss_ttt = engine(inputs)

    Reference:

        "ActMAD: Activation Matching to Align Distributions for Test-Time Training", M. Jehanzeb Mirza, Pol Jane Soneira, Wei Lin, Mateusz Kozinski, Horst Possegger, Horst Bischof

        Paper link: https://openaccess.thecvf.com/content/CVPR2023/papers/Mirza_ActMAD_Activation_Matching_To_Align_Distributions_for_Test-Time-Training_CVPR_2023_paper.pdf

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
    
    def compute_statistics(self, dataloader: DataLoader) -> None:
        """Extract and compute reference statistics for features.

        Args:
            dataloader (DataLoader): The dataloader used for extracting features. It can return tuples of tensors, with the first element expected to be the input tensor.

        Raises:
            ValueError: If the dataloader is empty or features have mismatched dimensions.
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