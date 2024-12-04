import torch 
from torchvision.transforms import functional as F
from torch_ttt.engine.base_engine import BaseEngine
from torch_ttt.engine_registry import EngineRegistry

# TODO: add cuda support
@EngineRegistry.register("ttt")
class TTTEngine(BaseEngine):

    def __init__(self, model, features_layer_name, angle_head=None, angle_criterion=None):
        super().__init__()
        self.model = model
        self.angle_head = angle_head
        self.angle_criterion = angle_criterion if angle_criterion else torch.nn.CrossEntropyLoss()
        self.features_layer_name = features_layer_name
        
       # Locate and store the reference to the target module
        self.target_module = None
        for name, module in model.named_modules():
            if name == features_layer_name:
                self.target_module = module
                break

        if self.target_module is None:
            raise ValueError(f"Module '{features_layer_name}' not found in the model.")
        
    def __call__(self, inputs):

        # has to dynamically register a hook to get the features and then remove it
        # need this for deepcopying the engine, see https://github.com/pytorch/pytorch/pull/103001
        features_hook = OutputHook()
        hook_handle = self.target_module.register_forward_hook(features_hook.hook)

        # Original forward pass, intact
        outputs = self.model(inputs)

        # FIXME: Add angles rotations for inputs
        # See original code: https://github.com/yueatsprograms/ttt_cifar_release/blob/acac817fb7615850d19a8f8e79930240c9afe8b5/main.py#L69
        rotated_inputs, rotation_labels = rotate_inputs(inputs)
        _ = self.model(rotated_inputs)
        features = features_hook.output
        hook_handle.remove()

        # Build angle head if not already built
        if self.angle_head is None:
            self.angle_head = build_angle_head(features)
        angles = self.angle_head(features)

        # Compute rotation loss
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
    rotated_image_90 = F.rotate(inputs, 90)
    rotated_image_180 = F.rotate(inputs, 180)
    rotated_image_270 = F.rotate(inputs, 270)
    batch_size = inputs.shape[0]
    inputs = torch.cat([inputs, rotated_image_90, rotated_image_180, rotated_image_270], dim=0)
    labels = [0] * batch_size + [1] * batch_size + [2] * batch_size + [3] * batch_size
    return inputs, torch.tensor(labels, dtype=torch.long)
    
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