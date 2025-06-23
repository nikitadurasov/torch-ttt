import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from copy import deepcopy
import math

from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["EATAEngine"]

@EngineRegistry.register("eata")
class EataEngine(BaseEngine):
    """**EATA**: Efficient Test-Time Adaptation without Forgetting.

    EATA improves test-time robustness by minimizing entropy on confident, diverse predictions, while preserving prior knowledge via Fisher regularization. Only BatchNorm affine parameters are updated.

    Args:
        model (torch.nn.Module): Model to be adapted at test-time.
        optimization_parameters (dict, optional): Optimizer configuration.
        e_margin (float): Entropy threshold to filter uncertain predictions.
        d_margin (float): Cosine similarity threshold to filter redundant samples.
        fisher_alpha (float): Weight for Fisher-based regularization.
        fishers (dict, optional): Precomputed Fisher information for regularization.

    :Example:

    .. code-block:: python

        from torch_ttt.engine.eata_engine import EataEngine

        model = MyModel()
        engine = EataEngine(model, {"lr": 1e-3})
        optimizer = torch.optim.SGD(engine.parameters(), lr=1e-3)

        # Training
        engine.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs, loss_ttt = engine(inputs)
            loss = criterion(outputs, labels) + alpha * loss_ttt
            loss.backward()
            optimizer.step()

        # Compute diagonal Fisher Information
        engine.compute_fishers(train_loader)

        # Inference
        engine.eval()
        for inputs, labels in test_loader:
            output, loss_ttt = engine(inputs)

    Reference:

        "Efficient Test-Time Model Adaptation without Forgetting",  
        Yuting Zhang, Srikrishna Karanam, Terrance E. Boult, Terrence Chen, Nuno Vasconcelos  
        ICML 2022

        Paper link: `PDF <https://arxiv.org/pdf/2204.02610.pdf>`_
    """

    def __init__(
        self,
        model: nn.Module,
        optimization_parameters: Optional[Dict[str, Any]] = None,
        
        # EATA specific parameters
        e_margin: float = math.log(1000) / 2 - 1,
        d_margin: float = 0.05,
        fisher_alpha: float = 2000.0,
        fishers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        super().__init__()
        self.model = model
        self.model.train()
        self._configure_bn()

        self.e_margin = e_margin
        self.d_margin = d_margin
        self.fisher_alpha = fisher_alpha
        self.fishers = fishers
        self.optimization_parameters = optimization_parameters or {}

        self.current_model_probs = None

    def _configure_bn(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.requires_grad_(True)
                module.track_running_stats = False
            else:
                for param in module.parameters(recurse=False):
                    param.requires_grad = False

    def ttt_forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass and test-time loss computation.

        Args:
            inputs (torch.Tensor): Input batch.

        Returns:
            Returns the current prediction, and the filtered and regularized adaptation loss.
        """
        outputs = self.model(inputs)
        loss, updated_probs = self.compute_loss(inputs, outputs)
        self.current_model_probs = updated_probs
        return outputs, loss

    def compute_loss(self, x, outputs):
        entropy = self.softmax_entropy(outputs)
        reliable_mask = entropy < self.e_margin
        outputs_reliable = outputs[reliable_mask]
        entropy_reliable = entropy[reliable_mask]

        # Filter redundant samples
        if self.current_model_probs is not None and outputs_reliable.size(0) > 0:
            cos_sim = F.cosine_similarity(
                self.current_model_probs.unsqueeze(0),
                outputs_reliable.softmax(dim=1), dim=1
            )
            non_redundant_mask = torch.abs(cos_sim) < self.d_margin
            outputs_final = outputs_reliable[non_redundant_mask]
            entropy_final = entropy_reliable[non_redundant_mask]
        else:
            outputs_final = outputs_reliable
            entropy_final = entropy_reliable

        updated_probs = self.update_model_probs(self.current_model_probs, outputs_final.softmax(dim=1))

        if entropy_final.size(0) == 0:
            loss = entropy_final.mean() * 0
        else:
            coeff = 1 / torch.exp(entropy_final.detach() - self.e_margin)
            loss = (entropy_final * coeff).mean()

        if self.fishers is not None:
            ewc_loss = 0.0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    fisher_val, init_val = self.fishers[name]
                    ewc_loss += self.fisher_alpha * (fisher_val * (param - init_val).pow(2)).sum()
            loss += ewc_loss

        return loss, updated_probs

    def update_model_probs(self, current_probs, new_probs):
        if new_probs.size(0) == 0:
            return current_probs
        if current_probs is None:
            return new_probs.mean(0).detach()
        return (0.9 * current_probs + 0.1 * new_probs.mean(0).detach())

    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(x, dim=1)
        log_probs = F.log_softmax(x, dim=1)
        return -(probs * log_probs).sum(dim=1)

    def compute_fishers(
        self,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: Optional[nn.Module] = None,
        lr: float = 0.001,

    ):
        """
        Estimate diagonal Fisher Information and store to `self.fishers`.
        
        Args:
            data_loader: DataLoader over source data.
            loss_fn: Loss to use for computing Fisher (default: CrossEntropy).
            lr: Learning rate for dummy optimizer (used to collect parameters).
            logger: Optional logger to print progress.
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        self.model = self._configure_bn(self.model)

        params, _ = self.collect_bn_params()
        optimizer = torch.optim.SGD(params, lr)

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss().to(device)

        fishers = {}

        for sample in enumerate(data_loader):
            images = sample[0] if isinstance(sample, (list, tuple)) else sample
            images = images.to(device, non_blocking=True)
            outputs = self.model(images)
            targets = outputs.argmax(dim=1)  # pseudo-labels as in original EATA
            loss = loss_fn(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_sq = param.grad.detach().clone() ** 2
                    if name in fishers:
                        fishers[name][0] += grad_sq
                    else:
                        fishers[name] = [grad_sq, param.detach().clone()]
            
            optimizer.zero_grad()

        # Normalize fishers
        for name in fishers:
            fishers[name][0] /= len(data_loader)

        self.fishers = fishers
        del optimizer

    def collect_bn_params(self):
        """Collects affine parameters (scale and shift) from all BatchNorm2d layers.

        Returns:
            Tuple[List[torch.nn.Parameter], List[str]]: 
                - List of parameters (`weight` and `bias`) to adapt during test-time training.
                - Corresponding parameter names for tracking or regularization.
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names