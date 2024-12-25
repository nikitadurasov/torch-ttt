import torch
from torch_ttt.loss.base_loss import BaseLoss
from torch_ttt.loss_registry import LossRegistry


@LossRegistry.register("weights_magnitude")
class WeightsMagnitudeLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.quantile = 0.95

    def __call__(self, model, inputs):
        # Step 2: Collect all model weights
        all_weights = []
        for param in model.parameters():
            if param.requires_grad:  # Focus only on trainable parameters
                all_weights.append(param.view(-1))  # Flatten weights into a 1D tensor

        # Concatenate all weights into a single tensor
        all_weights = torch.cat(all_weights)

        # Step 3: Compute the top `quantile` values
        quantile_value = torch.quantile(all_weights.abs(), self.quantile)
        top_quantile_weights = all_weights[all_weights.abs() >= quantile_value]

        # Step 4: Compute the average of top quantile weights
        weight_loss = top_quantile_weights.mean()

        # Return the combined loss
        return weight_loss
