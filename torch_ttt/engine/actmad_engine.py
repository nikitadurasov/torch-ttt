import torch
from typing import List, Dict, Any, Tuple, Union
from contextlib import contextmanager

from torch.utils.data import DataLoader
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

__all__ = ["ActMADEngine"]


@EngineRegistry.register("actmad")
class ActMADEngine(BaseEngine):
    """**ActMAD** approach: multi-level pixel-wise feature alignment.

    ActMAD adapts models at test-time by aligning activation statistics (means and variances)
    of the test inputs to those from clean training data, across multiple layers of the network. It requires no labels or auxiliary tasks, and is applicable to any architecture and task.
    
    Args:
        model (torch.nn.Module): Model to be trained with TTT.
        features_layer_names (List[str] | str): List of layer names to be used for feature alignment.
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

        Paper link: `PDF <https://proceedings.neurips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html>`_
    """

    def __init__(
            self,
            model: torch.nn.Module,
            features_layer_names: Union[List[str], str],
            optimization_parameters: Dict[str, Any] = {},
    ):
        super().__init__()
        self.model = model
        self.features_layer_names = features_layer_names
        self.optimization_parameters = optimization_parameters

        if isinstance(features_layer_names, str):
            self.features_layer_names = [features_layer_names]

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
            
        self.reference_mean = None
        self.reference_var = None

    def ttt_forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            The current model prediction and the alignment loss based on activation statistics.
        """
        with self.__capture_hook() as features_hooks:
            outputs = self.model(inputs)
            features = [hook.output for hook in features_hooks]

        # don't compute loss during training
        if self.training:
            return outputs, 0
        
        if self.reference_var is None or self.reference_mean is None:
            raise ValueError(
                "Reference statistics are not computed. Please call `compute_statistics` method."
            )

        l1_loss = torch.nn.L1Loss(reduction='mean')
        features_means = [torch.mean(feature, dim=0) for feature in features]
        features_vars = [torch.var(feature, dim=0) for feature in features]

        loss = 0
        for i in range(len(self.target_modules)):
            print(features_means[i].device, self.reference_mean[i].device)
            loss += l1_loss(features_means[i], self.reference_mean[i])
            loss += l1_loss(features_vars[i], self.reference_var[i])

        return outputs, loss
    
    def compute_statistics(self, dataloader: DataLoader) -> None:
        """Extract and compute reference statistics for features.

        Args:
            dataloader (DataLoader): The dataloader used for extracting features. It can return tuples of tensors, with the first element expected to be the input tensor.

        Raises:
            ValueError: If the dataloader is empty or features have mismatched dimensions.
        """
        
        self.model.eval()
        feat_stack = [[] for _ in self.target_modules]

        # TODO: compute variance in more memory efficient way
        with torch.no_grad():
            device = next(self.model.parameters()).device
            for sample in dataloader:
                if len(sample) < 1:
                    raise ValueError("Dataloader returned an empty batch.")
                
                inputs = sample[0].to(device)
                with self.__capture_hook() as features_hooks:
                    _ = self.model(inputs)
                    features = [hook.output.cpu() for hook in features_hooks]

                for i, feature in enumerate(features):
                    feat_stack[i].append(feature)

        # Compute mean and variance 
        self.reference_mean = [torch.mean(torch.cat(feat), dim=0).to(device) for feat in feat_stack]
        self.reference_var = [torch.var(torch.cat(feat), dim=0).to(device) for feat in feat_stack]

    @contextmanager
    def __capture_hook(self):
        """Context manager to capture features via a forward hook."""
    
        class OutputHook:
            def __init__(self):
                self.output = None

            def hook(self, module, input, output):
                self.output = output

        hook_handels = []
        features_hooks = []
        for module in self.target_modules:
            hook = OutputHook()
            features_hooks.append(hook)
            hook_handle = module.register_forward_hook(hook.hook)
            hook_handels.append(hook_handle)

        try:
            yield features_hooks
        finally:
            for hook in hook_handels:
                hook.remove()