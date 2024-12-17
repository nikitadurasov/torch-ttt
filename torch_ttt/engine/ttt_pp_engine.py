import torch
import random
from typing import Tuple, Optional, Callable
from contextlib import contextmanager

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry
from torch_ttt.loss.contrastive_loss import ContrastiveLoss

__all__ = ["TTTPPEngine"]

# TODO: finish this class
@EngineRegistry.register("ttt_pp")
class TTTPPEngine(BaseEngine):
    """ **TTT++** approach: feature alignment-based + SimCLR loss.

    Args:
        model (torch.nn.Module): Model to be trained with TTT.
        features_layer_name (str): The name of the layer from which the features are extracted.
        contrastive_head (torch.nn.Module, optional): The head that is used for SimCLR's Loss.
        contrastive_criterion (torch.nn.Module, optional): The loss function used for SimCLR. 
        contrastive_transform (callable): A transformation or a composition of transformations applied to the input images to generate augmented views for contrastive learning.

    Reference:
        "TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?", Yuejiang Liu, Parth Kothari, Bastien van Delft, Baptiste Bellot-Gurlet, Taylor Mordan, Alexandre Alahi

        Paper link: https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html
    
    """
    def __init__(
            self,
            model: torch.nn.Module,
            features_layer_name: str, 
            contrastive_head: torch.nn.Module = None,
            contrastive_criterion: torch.nn.Module = None,
            contrastive_transform: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.model = model
        self.features_layer_name = features_layer_name
        self.contrastive_head = contrastive_head
        self.contrastive_criterion = contrastive_criterion if contrastive_criterion else ContrastiveLoss()

        # Locate and store the reference to the target module
        self.target_module = None
        for name, module in model.named_modules():
            if name == features_layer_name:
                self.target_module = module
                break

        if self.target_module is None:
            raise ValueError(f"Module '{features_layer_name}' not found in the model.")

        if contrastive_transform is None:
            self.contrastive_transform = transforms.Compose(
                [
                    RandomResizedCrop(scale=(0.2, 1.)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),
                    transforms.RandomHorizontalFlip()
                ]
            )
        
    def forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model. 
    
        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            Returns the model prediction and TTT++ loss value.
        """
        contrastive_inputs = torch.cat(
            [
                self.contrastive_transform(inputs), 
                self.contrastive_transform(inputs)
            ], 
            dim=0
        )

        with self.__capture_hook() as features_hook:
            outputs = self.model(contrastive_inputs)
            features = features_hook.output

        features = features.view(2, len(inputs), -1).transpose(0, 1)
        contrastive_loss = self.contrastive_criterion(features)

        return outputs[:len(inputs)], contrastive_loss

    def compute_statistics(self, dataloader: DataLoader) -> None:
        """Extract features from the target module."""
        pass

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

class RandomResizedCrop:
    def __init__(self, scale=(0.2, 1.0)):
        self.scale = scale

    def __call__(self, img):
        # Dynamically compute the crop size
        original_size = img.shape[-2:]  # H × W
        area = original_size[0] * original_size[1]
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(3/4, 4/3)

        h = int(round((target_area * aspect_ratio) ** 0.5))
        w = int(round((target_area / aspect_ratio) ** 0.5))

        if random.random() < 0.5:  # Randomly swap h and w
            h, w = w, h

        h = min(h, original_size[0])
        w = min(w, original_size[1])

        return F.resized_crop(img, top=0, left=0, height=h, width=w, size=original_size)
