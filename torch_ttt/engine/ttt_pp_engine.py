import torch
import random
from typing import Tuple, Optional, Callable
from contextlib import contextmanager
import statistics

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
        scale_cov (float): The scale factor for the covariance loss.
        scale_mu (float): The scale factor for the mean loss.
        scale_c_cov (float): The scale factor for the contrastive covariance loss.
        scale_c_mu (float): The scale factor for the contrastive mean loss.

    :Example:

    .. code-block:: python

        from torch_ttt.engine.ttt_pp_engine import TTTPPEngine

        model = MyModel()
        engine = TTTPPEngine(model, "fc1")

        # Training 
        engine.train()
        for inputs, labels in train_loader:
            ...

        engine.compute_statistics(train_loader)

        # Inference 
        engine.eval()
        for inputs, labels in test_loader:
            ...

    Reference:
        "TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?", Yuejiang Liu, Parth Kothari, Bastien van Delft, Baptiste Bellot-Gurlet, Taylor Mordan, Alexandre Alahi

        Paper link: https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html
    
    """
    def __init__(
            self,
            model: torch.nn.Module,
            features_layer_name: str, 
            contrastive_head: torch.nn.Module = None,
            contrastive_criterion: torch.nn.Module = ContrastiveLoss(),
            contrastive_transform: Optional[Callable] = None,
            scale_cov: float = 0.1,
            scale_mu: float = 0.1,
            scale_c_cov: float = 0.1,
            scale_c_mu: float = 0.1
    ) -> None:
        super().__init__()
        self.model = model
        self.features_layer_name = features_layer_name
        self.contrastive_head = contrastive_head
        self.contrastive_criterion = contrastive_criterion if contrastive_criterion else ContrastiveLoss()
        self.scale_cov = scale_cov
        self.scale_mu = scale_mu
        self.scale_c_cov = scale_c_cov
        self.scale_c_mu = scale_c_mu
        self.contrastive_transform = contrastive_transform

        self.reference_cov = None
        self.reference_mean = None
        self.reference_c_cov = None
        self.reference_c_mean = None

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

    def __build_contrastive_head(self, features) -> torch.nn.Module:
        """Build the angle head."""
        # See original implementation: https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/utils/test_helpers.py#L33C10-L33C39
        if len(features.shape) == 2:
            return torch.nn.Sequential(
                torch.nn.Linear(features.shape[1], 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 4)
            )
        
        raise ValueError("Features should be 2D tensor.")
        
    def forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model. 
    
        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            Returns the model prediction and TTT++ loss value.
        """

        # reset reference statistics during training
        if self.training:
            self.reference_cov = None
            self.reference_mean = None

        contrastive_inputs = torch.cat(
            [
                self.contrastive_transform(inputs), 
                self.contrastive_transform(inputs)
            ], 
            dim=0
        )

        # extract features for contrastive loss
        with self.__capture_hook() as features_hook:
            _ = self.model(contrastive_inputs)
            features = features_hook.output

        # Build angle head if not already built
        if self.contrastive_head is None:
            self.contrastive_head = self.__build_contrastive_head(features)

        contrasitve_features = self.contrastive_head(features)
        contrasitve_features = contrasitve_features.view(2, len(inputs), -1).transpose(0, 1)
        loss = self.contrastive_criterion(contrasitve_features)

        # make inference for a final prediction
        with self.__capture_hook() as features_hook:
            outputs = self.model(inputs)
            features = features_hook.output

        # compute alignment loss only during test
        if not self.training:

            if self.reference_cov is None or self.reference_mean is None:
                raise ValueError("Reference statistics are not computed. Please call `compute_statistics` method.")

            # compute features alignment loss
            cov_ext = self._covariance(features)
            mu_ext = features.mean(dim=0)

            d = self.reference_cov.shape[0]
            loss += self.scale_cov * (self.reference_cov - cov_ext).pow(2).sum() / (4. * d ** 2)
            loss += self.scale_mu * (self.reference_mean - mu_ext).pow(2).mean()            

            # compute contrastive features alignment loss
            c_features = self.contrastive_head(features)

            cov_ext = self._covariance(c_features)
            mu_ext = c_features.mean(dim=0)

            d = self.reference_c_cov.shape[0]
            loss += self.scale_c_cov * (self.reference_c_cov - cov_ext).pow(2).sum() / (4. * d ** 2)
            loss += self.scale_c_mu * (self.reference_c_mean - mu_ext).pow(2).mean()     

        return outputs, loss

    @staticmethod
    def _covariance(features):
        assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
        n = features.shape[0]
        tmp = torch.ones((1, n), device=features.device) @ features
        cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
        return cov

    def compute_statistics(self, dataloader: DataLoader) -> None:
        """Extract features from the target module."""

        self.model.eval()

        feat_stack = []
        c_feat_stack = []

        with torch.no_grad():
            for sample in dataloader:
                with self.__capture_hook() as features_hook:

                    inputs = sample[0]
                    _ = self.model(inputs)
                    feat = features_hook.output

                    # Build angle head if not already built
                    if self.contrastive_head is None:
                        self.contrastive_head = self.__build_contrastive_head(feat)

                    contrastive_feat = self.contrastive_head(feat)

                feat_stack.append(feat)
                c_feat_stack.append(contrastive_feat)

        # compute features statistics
        feat_all = torch.cat(feat_stack)
        feat_cov = self._covariance(feat_all)
        feat_mean = feat_all.mean(dim=0)
        
        self.reference_cov = feat_cov 
        self.reference_mean = feat_mean

        # compute contrastive features statistics
        feat_all = torch.cat(c_feat_stack)
        feat_cov = self._covariance(feat_all)
        feat_mean = feat_all.mean(dim=0)
        
        self.reference_c_cov = feat_cov
        self.reference_c_mean = feat_mean

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
