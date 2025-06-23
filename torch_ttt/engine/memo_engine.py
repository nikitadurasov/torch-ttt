import torch
import random
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Any, Tuple, List
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["MemoEngine"]

@EngineRegistry.register("memo")
class MemoEngine(BaseEngine):
    """**MEMO**: Test-Time Robustness via Augmentation.

    Applies multiple augmentations per test sample and adapts the model
    by minimizing the entropy of the average prediction across augmentations.

    Args:
        model (torch.nn.Module): The model to adapt.
        optimization_parameters (dict): Hyperparameters for adaptation.
        n_augmentations (int): Number of augmented views per input sample.

    :Example:

    .. code-block:: python

        from torch_ttt.engine.memo_engine import MemoEngine

        model = MyModel()
        engine = MemoEngine(model, {"lr": 1e-3}, n_augmentations=4)
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

        "Memo: Test-Time Robustness via Adaptation and Augmentation", 
        Marvin Zhang, Sergey Levine, Chelsea Finn

        Paper link: `PDF <https://proceedings.neurips.cc/paper_files/paper/2022/file/fc28053a08f59fccb48b11f2e31e81c7-Paper-Conference.pdf>`_
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimization_parameters: Dict[str, Any] = {},
        augmentations = None,
        n_augmentations: int = 8,
        prior_strength: float = 16
    ):
        super().__init__()
        self.model = model
        self.optimization_parameters = optimization_parameters
        self.augmentations = augmentations
        self.n_augmentations = n_augmentations
        self.prior_strength = prior_strength

        if augmentations is None:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                AddGaussianNoise(std=0.02, p=0.5),
                RandomGaussianBlur(kernel_size=3, sigma=(0.1, 1.5), p=0.5),
            ])  
        

        self.model.train()
        self._configure_bn()

    def _configure_bn(self):
        for module in self.model.modules():
            # check https://github.com/zhangmarvin/memo/blob/228b2908d271c954ef8bf19cf143ede3b2fa8e3e/imagenet-exps/utils/train_helpers.py#L56C36-L56C95
            if isinstance(module, torch.nn.BatchNorm2d):
                module.prior = float(self.prior_strength) / float(self.prior_strength + 1)
                module.forward = _modified_bn_forward

    def _marginal_entropy(self, outputs):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1).mean(), avg_logits

    def ttt_forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs MEMO test-time forward pass.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: (final prediction logits, adaptation loss)
        """

        augmented_inputs = [self.augmentations(inputs) for _ in range(self.n_augmentations)]
        augmented_inputs = torch.stack(augmented_inputs)
        augmented_inputs = augmented_inputs.view(-1, *augmented_inputs.shape[2:])
        logits = self.model(augmented_inputs)
        logits = logits.view(self.n_augmentations, -1, logits.shape[-1])
        loss, logits = self._marginal_entropy(logits)

        print(loss, loss.shape)
        return logits, loss
    
def _modified_bn_forward(self, input):
    # https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py#L175
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    # update est_mean and est_var with the current statistics
    torch.nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return torch.nn.functional.batch_norm(
        input, running_mean, running_var, 
        self.weight, self.bias, False, 0, self.eps
    )

# Optional noise injection
class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.03, p=0.5):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, x):
        if self.training and random.random() < self.p:
            return x + torch.randn_like(x) * self.std
        return x

# Optional Gaussian blur (for tensors)
class RandomGaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def forward(self, x):
        if self.training and random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            return transforms.functional.gaussian_blur(x, self.kernel_size, sigma)
        return x