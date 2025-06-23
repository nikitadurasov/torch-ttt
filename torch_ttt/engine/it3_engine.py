# torch_ttt/engine/ittt_engine.py
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Callable, Dict, Any, Tuple, Optional

from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["IT3Engine"]


@EngineRegistry.register("ittt")
class IT3Engine(BaseEngine):
    r"""**IT³**: Idempotent Test-Time Training

    A domain-agnostic **test-time training** method that adapts model predictions 
    by enforcing idempotence—ensuring that repeated applications of the model 
    yield consistent outputs.

    Args:
        model (torch.nn.Module): A pre-trained model to be adapted with IT3.
        features_layer_name (str): Name of the layer to inject features into during TTT.
        embeder (torch.nn.Module, optional): Module to embed the initial prediction. If None, it will be created dynamically.
        combine_fn (Callable, optional): Function to combine the hidden features and embedding. If None, uses broadcast addition.
        distance_fn (Callable, optional): Distance function used to compare successive predictions. Defaults to MSE loss.
        optimization_parameters (dict): Parameters controlling adaptation (e.g., learning rate, optimizer choice).

    Warning:
        The module with name :attr:`features_layer_name` must exist in the model.

    Note:
        If `embeder` and `combine_fn` are not provided, they are constructed on-the-fly 
        to match the dimensions of predictions and the injection layer (supporting 2D/4D tensors).

    Example:

    .. code-block:: python

        from torch_ttt.engine.ittt_engine import IT3Engine

        model = MyModel()
        engine = IT3Engine(model, "encoder")
        optimizer = torch.optim.Adam(engine.parameters(), lr=1e-4)

        # Training
        engine.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs, loss_ttt = engine(inputs, target=labels)
            loss = criterion(outputs, labels) + alpha * loss_ttt
            loss.backward()
            optimizer.step()

        # Inference
        engine.eval()
        for inputs in test_loader:
            outputs, loss_ttt = engine(inputs)

    Reference:

        "IT³: Idempotent Test-Time Training", Nikita Durasov, Assaf Shocher, Doruk Oner, 
        Gal Chechik, Alexei A. Efros, Pascal Fua. ICML 2025.

        Paper link: `PDF <https://arxiv.org/abs/2410.04201>`_
    """

    def __init__(
        self,
        model: nn.Module,
        features_layer_name: str,
        embeder: Optional[nn.Module] = None,
        combine_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        distance_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        optimization_parameters: Dict[str, Any] = {},
    ):
        super().__init__()
        self.model = model
        self.features_layer_name = features_layer_name
        self.optimization_parameters = optimization_parameters

        self.embeder = embeder            # None → built lazily
        self.combine_fn = combine_fn      # None → built lazily
        self.distance_fn = distance_fn or (lambda y1, y0:
                                           torch.nn.functional.mse_loss(y1, y0))

        self.target_module = self._find_module(features_layer_name)
        if self.target_module is None:
            raise ValueError(f"Module '{features_layer_name}' not found in the model.")

        self._defaults_ready = embeder is not None and combine_fn is not None

    # ------------------------------------------------------------------ #
    #  utilities
    # ------------------------------------------------------------------ #
    def _find_module(self, name: str):
        for n, m in self.model.named_modules():
            if n == name:
                return m
        return None

    def _broadcast_add(self, feat: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Element-wise addition with automatic broadcasting."""
        # If `emb` has fewer dimensions, add singleton dimensions on the right
        while emb.dim() < feat.dim():
            emb = emb.unsqueeze(-1)
        return feat + emb

    def _build_defaults(self,
                        y0: torch.Tensor,
                        feat: torch.Tensor,
                        device: torch.device):
        """Constructs a linear/conv embedder and combine_fn based on tensor shape."""
        in_ch = y0.size(1) if y0.dim() >= 2 else y0.size(-1)
        out_ch = feat.size(1) if feat.dim() >= 2 else feat.size(-1)

        if y0.dim() == 2:                                # (B, C)
            self.embeder = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.LeakyReLU(),
                nn.Linear(out_ch, out_ch),
                nn.LeakyReLU(),
            ).to(device)

        elif y0.dim() == 4:                              # (B, C, H, W)
            self.embeder = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=1),
                nn.LeakyReLU(),
            )

        else:                                            # fallback to identity
            self.embeder = nn.Identity()

        self.combine_fn = self._broadcast_add
        self._defaults_ready = True

    # ------------------------------------------------------------------ #
    #  feature injection hook
    # ------------------------------------------------------------------ #
    @contextmanager
    def _inject(self, emb: torch.Tensor):
        def _hook(_, __, output):
            return self.combine_fn(output, emb)

        h = self.target_module.register_forward_hook(_hook)
        try:
            yield
        finally:
            h.remove()

    # ------------------------------------------------------------------ #
    #  main forward
    # ------------------------------------------------------------------ #
    def ttt_forward(self, inputs, target=None):
        """Two forward passes; gradients enabled to allow adaptation."""
        x = inputs if torch.is_tensor(inputs) else inputs["x"]

        # ---------- First forward pass and feature capture ----------
        bucket = {}
        handle = self.target_module.register_forward_hook(lambda _, __, o: bucket.update(feat=o))
        y0 = self.model(x)  # gradients ON
        handle.remove()
        feat = bucket["feat"]

        if not self._defaults_ready:
            self._build_defaults(y0, feat, x.device)

        if target is not None:
            emb = self.embeder(target)
        else:
            emb = self.embeder(y0)  # gradients remain enabled
        # To block gradients to the embedder, uncomment:
        # emb = emb.detach()

        # ---------- Second forward pass with feature injection ----------
        with self._inject(emb):
            y1 = self.model(x)

        if target is not None:
            loss = self.distance_fn(y1, target)
        else:
            loss = self.distance_fn(y1, y0)

        return y0, loss
