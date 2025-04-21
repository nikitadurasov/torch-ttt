from contextlib import contextmanager
import torch
from copy import deepcopy
from typing import Tuple, Dict, Any, List
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["MaskedTTTEngine"]


@EngineRegistry.register("masked_ttt")
class MaskedTTTEngine(BaseEngine):
    r"""Masked token prediction-based **test-time training** engine.

    This engine performs masked language modeling (MLM) as a self-supervised auxiliary task during inference.
    It randomly masks input tokens (except those in `skip_tokens`) and trains the model to predict them using intermediate features.

    Args:
        model (torch.nn.Module): The model with a Transformer block to adapt at test time.
        mask_token_id (int): The token ID used for masking input tokens.
        features_layer_name (str): Name of the intermediate layer from which logits are extracted.
        mask_prob (float, optional): Probability of masking each token. Default is 0.15.
        skip_tokens (list, optional): List of token IDs to skip when applying the mask.

    Warning:
        The module with the name specified by :attr:`features_layer_name` must exist within the model.

    :Example:

    .. code-block:: python

        from torch_ttt.engine.masked_ttt_engine import MaskedTTTEngine

        model = MyTransformerModel()
        engine = MaskedTTTEngine(model, mask_token_id=103, features_layer_name="encoder.layer.11.output")

        optimizer = torch.optim.Adam(engine.parameters(), lr=1e-4)

        # Training with TTT (Important! MaskedTTTEngine can be applied to already pretrained models)
        engine.train()
        for batch in test_loader:
            optimizer.zero_grad()
            outputs, loss_ttt = engine.ttt_forward(batch)
            loss_ttt.backward()
            optimizer.step()

        # Inference
        engine.eval()
        for batch in test_loader:
            outputs, loss_ttt = engine.ttt_forward(batch)

    Reference:
        This approach is inspired by masked language modeling and related TTT methods, e.g., MAE and self-supervised transformers.
    """


    def __init__(
            self, 
            model: torch.nn.Module, 
            mask_token_id: int, 
            features_layer_name: str, 
            mask_prob: float = 0.15, 
            skip_tokens: List[int] =[],
            optimization_parameters: Dict[str, Any] = {},
        ):

        super().__init__()
        self.model = model
        self.mask_token_id = mask_token_id
        self.features_layer_name = features_layer_name
        self.mask_prob = mask_prob
        self.skip_tokens = skip_tokens
        self.optimization_parameters = optimization_parameters
        self.loss = torch.nn.CrossEntropyLoss()

        # Locate and store the reference to the target module
        self.target_module = None
        for name, module in model.named_modules():
            if name == features_layer_name:
                self.target_module = module
                break

        if self.target_module is None:
            raise ValueError(f"Module '{features_layer_name}' not found in the model.")


    def _mask_tokens(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len)
            mask: (batch_size, seq_len)
        """
        masksed_inputs_ids = input_ids.clone()
        device = masksed_inputs_ids.device
        mask = torch.rand(masksed_inputs_ids.shape).to(device) < self.mask_prob
        for skip_token in self.skip_tokens:
            mask = mask & (masksed_inputs_ids != skip_token)
        masksed_inputs_ids[mask] = self.mask_token_id
        return masksed_inputs_ids, mask

    def ttt_forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
      # TODO: don't make the copy
      inputs_copy = deepcopy(inputs)
      inputs_idx = inputs_copy["input_ids"]
      masked_inputs_idx, mask = self._mask_tokens(inputs_idx)
      inputs_copy["input_ids"] = masked_inputs_idx

      with self.__capture_hook() as features_hook:
          outputs = self.model(**inputs_copy)
          features = features_hook.output

      loss = self.loss(
          features[mask],
          inputs_idx[mask]
      )
        
      if mask.sum() == 0:
          loss = features.mean() * 0 # return differentiable non-informative 0 loss
          
      return outputs, loss

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