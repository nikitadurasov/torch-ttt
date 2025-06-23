import torch
from typing import Tuple, Optional, Callable, Dict, Any
from contextlib import contextmanager

from torchvision import transforms
from torch.utils.data import DataLoader
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry
from torch_ttt.loss.contrastive_loss import ContrastiveLoss
from torch_ttt.utils.augmentations import RandomResizedCrop

__all__ = ["TTTPPEngine"]


# TODO: finish this class
@EngineRegistry.register("ttt_pp")
class TTTPPEngine(BaseEngine):
    """**TTT++** approach: feature alignment-based + SimCLR loss.

    A test-time training method that builds on TTT by enforcing consistency across both standard and contrastive features. It adapts the model during inference through SimCLR-based contrastive loss and alignment of features to training-time statistics.

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
        optimization_parameters (dict): The optimization parameters for the engine.
        
    Warning:
        The module with the name :attr:`features_layer_name` should be present in the model.

    :Example:

    .. code-block:: python

        from torch_ttt.engine.ttt_pp_engine import TTTPPEngine

        model = MyModel()
        engine = TTTPPEngine(model, "fc1")
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

        "TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?", Yuejiang Liu, Parth Kothari, Bastien van Delft, Baptiste Bellot-Gurlet, Taylor Mordan, Alexandre Alahi

        Paper link: `PDF <https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html>`_
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
        scale_c_mu: float = 0.1,
        optimization_parameters: Dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.model = model
        self.features_layer_name = features_layer_name
        self.contrastive_head = contrastive_head
        self.contrastive_criterion = (
            contrastive_criterion if contrastive_criterion else ContrastiveLoss()
        )
        self.scale_cov = scale_cov
        self.scale_mu = scale_mu
        self.scale_c_cov = scale_c_cov
        self.scale_c_mu = scale_c_mu
        self.contrastive_transform = contrastive_transform

        self.reference_cov = None
        self.reference_mean = None
        self.reference_c_cov = None
        self.reference_c_mean = None

        self.optimization_parameters = optimization_parameters

        # Locate and store the reference to the target module
        self.target_module = None
        for name, module in model.named_modules():
            if name == features_layer_name:
                self.target_module = module
                break

        if self.target_module is None:
            raise ValueError(f"Module '{features_layer_name}' not found in the model.")

        # Validate that the target module is a Linear layer
        if not isinstance(self.target_module, torch.nn.Linear):
            raise TypeError(
                f"Module '{features_layer_name}' is expected to be of type 'torch.nn.Linear', "
                f"but found type '{type(self.target_module).__name__}'."
            )

        if contrastive_transform is None:
            # default SimCLR augmentation
            self.contrastive_transform = transforms.Compose(
                [
                    RandomResizedCrop(scale=(0.2, 1.0)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),
                    transforms.RandomHorizontalFlip(),
                ]
            )

    def __build_contrastive_head(self, features) -> torch.nn.Module:
        """Build the angle head."""
        device = next(self.model.parameters()).device
        if len(features.shape) == 2:
            return torch.nn.Sequential(
                torch.nn.Linear(features.shape[1], 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 16),
            ).to(device)

        raise ValueError("Features should be 2D tensor.")

    def ttt_forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
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
            self.reference_c_cov = None
            self.reference_c_mean = None

        contrastive_inputs = torch.cat(
            [self.contrastive_transform(inputs), self.contrastive_transform(inputs)], dim=0
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
            if (
                self.reference_cov is None
                or self.reference_mean is None
                or self.reference_c_cov is None
                or self.reference_c_mean is None
            ):
                raise ValueError(
                    "Reference statistics are not computed. Please call `compute_statistics` method."
                )

            # compute features alignment loss
            cov_ext = self.__covariance(features)
            mu_ext = features.mean(dim=0)

            d = self.reference_cov.shape[0]

            loss += self.scale_cov * (self.reference_cov - cov_ext).pow(2).sum() / (4.0 * d**2)
            loss += self.scale_mu * (self.reference_mean - mu_ext).pow(2).mean()

            # compute contrastive features alignment loss
            c_features = self.contrastive_head(features)

            cov_ext = self.__covariance(c_features)
            mu_ext = c_features.mean(dim=0)

            d = self.reference_c_cov.shape[0]
            loss += self.scale_c_cov * (self.reference_c_cov - cov_ext).pow(2).sum() / (4.0 * d**2)
            loss += self.scale_c_mu * (self.reference_c_mean - mu_ext).pow(2).mean()

        return outputs, loss

    @staticmethod
    def __covariance(features):
        """Legacy wrapper to maintain compatibility in the engine."""
        from torch_ttt.utils.math import compute_covariance

        return compute_covariance(features, dim=0)

    def compute_statistics(self, dataloader: DataLoader) -> None:
        """Extract and compute reference statistics for features and contrastive features.

        Args:
            dataloader (DataLoader): The dataloader used for extracting features. It can return tuples of tensors, with the first element expected to be the input tensor.

        Raises:
            ValueError: If the dataloader is empty or features have mismatched dimensions.
        """

        self.model.eval()

        feat_stack = []
        c_feat_stack = []

        with torch.no_grad():
            device = next(self.model.parameters()).device
            for sample in dataloader:
                if len(sample) < 1:
                    raise ValueError("Dataloader returned an empty batch.")

                inputs = sample[0].to(device)
                with self.__capture_hook() as features_hook:
                    _ = self.model(inputs)
                    feat = features_hook.output

                    # Initialize contrastive head if not already initialized
                    if self.contrastive_head is None:
                        self.contrastive_head = self.__build_contrastive_head(feat)

                    # Compute contrastive features
                    contrastive_feat = self.contrastive_head(feat)

                feat_stack.append(feat.cpu())
                c_feat_stack.append(contrastive_feat.cpu())

        # compute features statistics
        feat_all = torch.cat(feat_stack)
        feat_cov = self.__covariance(feat_all)
        feat_mean = feat_all.mean(dim=0)

        self.reference_cov = feat_cov.to(device)
        self.reference_mean = feat_mean.to(device)

        # compute contrastive features statistics
        feat_all = torch.cat(c_feat_stack)
        feat_cov = self.__covariance(feat_all)
        feat_mean = feat_all.mean(dim=0)

        self.reference_c_cov = feat_cov.to(device)
        self.reference_c_mean = feat_mean.to(device)

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
