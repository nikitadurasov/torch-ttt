import torch
from contextlib import contextmanager
from typing import Tuple, Dict, Any
from torchvision.transforms import functional as F
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["TTTEngine"]


@EngineRegistry.register("ttt")
class TTTEngine(BaseEngine):
    r"""Original image rotation-based **test-time training** approach.

    A test-time training method that improves model robustness by solving an auxiliary rotation prediction task during inference, encouraging better feature alignment under distribution shifts.

    Args:
        model (torch.nn.Module): Model to be trained with TTT.
        features_layer_name (str): The name of the layer from which the features are extracted.
        angle_head (torch.nn.Module, optional): The head that predicts the rotation angles.
        angle_criterion (torch.nn.Module, optional): The loss function for the rotation angles.
        optimization_parameters (dict): The optimization parameters for the engine.

    Warning:
        The module with the name :attr:`features_layer_name` should be present in the model.

    Note:
        :attr:`angle_head` and :attr:`angle_criterion` are optional arguments and can be user-defined. If not provided, the default shallow head and the :meth:`torch.nn.CrossEntropyLoss()` loss function are used.

    Note:
        The original `TTT <https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/utils/rotation.py#L27>`_ implementation uses a four-class classification task, corresponding to image rotations of 0°, 90°, 180°, and 270°.

    :Example:

    .. code-block:: python

        from torch_ttt.engine.ttt_engine import TTTEngine

        model = MyModel()
        engine = TTTEngine(model, "fc1")
        optimizer = torch.optim.Adam(engine.parameters(), lr=1e-4)

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

        "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts", Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, Moritz Hardt

        Paper link: `PDF <http://proceedings.mlr.press/v119/sun20b/sun20b.pdf>`_
    """

    def __init__(
        self,
        model: torch.nn.Module,
        features_layer_name: str,
        angle_head: torch.nn.Module = None,
        angle_criterion: torch.nn.Module = None,
        optimization_parameters: Dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.model = model
        self.angle_head = angle_head
        self.angle_criterion = angle_criterion if angle_criterion else torch.nn.CrossEntropyLoss()
        self.features_layer_name = features_layer_name
        self.optimization_parameters = optimization_parameters

        # Locate and store the reference to the target module
        self.target_module = None
        for name, module in model.named_modules():
            if name == features_layer_name:
                self.target_module = module
                break

        if self.target_module is None:
            raise ValueError(f"Module '{features_layer_name}' not found in the model.")

    def ttt_forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            Returns the current model prediction and rotation loss value.
        """

        # has to dynamically register a hook to get the features and then remove it
        # need this for deepcopying the engine, see https://github.com/pytorch/pytorch/pull/103001
        with self.__capture_hook() as features_hook:
            # Original forward pass, intact
            outputs = self.model(inputs)

            # See original code: https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/main.py#L69
            rotated_inputs, rotation_labels = self.__rotate_inputs(inputs)
            _ = self.model(rotated_inputs)
            features = features_hook.output

        # Build angle head if not already built
        if self.angle_head is None:
            self.angle_head = self.__build_angle_head(features)

        # move angle head to the same device as the features
        self.angle_head.to(features.device)
        angles = self.angle_head(features)

        # Compute rotation loss
        rotation_loss = self.angle_criterion(angles, rotation_labels)
        return outputs, rotation_loss

    # Follow this code (expand case): https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/utils/rotation.py#L27
    def __rotate_inputs(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate the input images by 0, 90, 180, and 270 degrees."""
        device = next(self.model.parameters()).device
        rotated_image_90 = F.rotate(inputs, 90)
        rotated_image_180 = F.rotate(inputs, 180)
        rotated_image_270 = F.rotate(inputs, 270)
        batch_size = inputs.shape[0]
        inputs = torch.cat([inputs, rotated_image_90, rotated_image_180, rotated_image_270], dim=0)
        labels = [0] * batch_size + [1] * batch_size + [2] * batch_size + [3] * batch_size
        return inputs.to(device), torch.tensor(labels, dtype=torch.long).to(device)

    def __build_angle_head(self, features) -> torch.nn.Module:
        """Build the angle head."""
        device = next(self.model.parameters()).device

        # See original implementation: https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/utils/test_helpers.py#L33C10-L33C39
        if len(features.shape) == 2:
            return torch.nn.Sequential(
                torch.nn.Linear(features.shape[1], 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 4),
            ).to(device)

        # See original implementation: https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/models/SSHead.py#L29
        elif len(features.shape) == 4:
            return torch.nn.Sequential(
                torch.nn.Conv2d(features.shape[1], 16, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 4, 3),
                torch.nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
                torch.nn.Flatten(),
            ).to(device)

        elif len(features.shape) == 5:  # For 3D inputs (batch, channels, depth, height, width)
            return torch.nn.Sequential(
                torch.nn.Conv3d(features.shape[1], 16, kernel_size=3),
                torch.nn.ReLU(),
                torch.nn.Conv3d(16, 4, kernel_size=3),
                torch.nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global Average Pooling
                torch.nn.Flatten(),
            ).to(device)

        raise ValueError("Invalid input tensor shape.")

    @contextmanager
    def __capture_hook(self):
        """Context manager to capture features via a forward hook."""

        class OutputHook:
            def __init__(self):
                self.output = None

            def hook(self, module, input, output):
                self.output = output

        features_hook = OutputHook()
        hook_handle = self.target_module.register_forward_hook(features_hook.hook)

        try:
            yield features_hook
        finally:
            hook_handle.remove()
