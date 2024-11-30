import torch 
import torch.nn.functional as F
from torch_ttt.core.engine.base_engine import BaseEngine
from torch_ttt.core.engine_registry import EngineRegistry

# TODO: add cuda support
@EngineRegistry.register("ttt")
class TTTEngine(BaseEngine):

    def __init__(self, model, features_layer_name, angle_head=None, angle_criterion=None):
        super().__init__()
        self.model = model
        self.angle_head = angle_head
        self.angle_criterion = angle_criterion if angle_criterion else torch.nn.CrossEntropyLoss()
        self.features_layer_name = features_layer_name
        
        # Register the hook to track the features 
        self.features_hook = OutputHook()
        target_layer_name = features_layer_name 
        for name, module in model.named_modules():
            if name == target_layer_name:
                module.register_forward_hook(self.features_hook.hook)

    def __call__(self, inputs):
        # Original forward pass, intact
        outputs = self.model(inputs)

        # FIXME: Add angles rotations for inputs
        # See original code: https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/main.py#L69
        rotated_inputs, rotation_labels = rotate_inputs(inputs)

        _ = self.model(rotated_inputs)
        features = self.features_hook.output

        if self.angle_head is None:
            self.angle_head = build_angle_head(features)
        angles = self.angle_head(features)
        rotation_loss = self.angle_criterion(angles, rotation_labels)
        return outputs, rotation_loss

class OutputHook:
    def __init__(self):
        self.output = None

    def hook(self, module, input, output):
        self.output = output

# TODO [P0]: implement rotations function
# Follow this code (expand case): https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/utils/rotation.py#L27
def rotate_inputs(inputs):
    batch_size = inputs.shape[0]
    return inputs, torch.randint(4, (batch_size,), dtype=torch.long)
    
def build_angle_head(features):

    # FIXME: align with the original architechture
    # See original implementation: https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/utils/test_helpers.py#L33C10-L33C39
    if len(features.shape) == 2:
        return torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4)
        )
    
    # FIXME: align with the original architechture
    # See original implementation: https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/models/SSHead.py#L29
    elif len(features.shape) == 4:
        return torch.nn.Sequential(
            torch.nn.Conv2d(features.shape[1], 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 4, 3),
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            torch.nn.Flatten() 
        )
    
    # If the input is a 3D tensor
    # TODO: Optimize this function for larger datasets
    elif len(features.shape) == 5:
        raise NotImplementedError("3D tensor case will be implemented later.")
    
    raise ValueError("Invalid input tensor shape.")