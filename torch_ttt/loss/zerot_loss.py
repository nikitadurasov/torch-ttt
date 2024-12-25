import torch
from torch_ttt.loss.base_loss import BaseLoss
from torch_ttt.loss_registry import LossRegistry


@LossRegistry.register("zerot")
class ZeroTrainLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.quantile = 0.95

    def __call__(self, model, inputs):
        N = len(list(model.named_parameters()))

        importance_dict = compute_weight_importance(model, inputs, N)

        # Calculate the top s% mean importance as the loss
        loss = top_s_percent_mean(importance_dict, self.quantile)

        # Return the combined loss
        return loss


def top_s_percent_mean(importance_dict, s):
    """
    Compute the mean of the top s% of the importance values.

    Args:
        importance_dict (dict): A dictionary mapping layer names to weight importance tensors.
        s (float): The percentage (0-100) of top importance values to consider.

    Returns:
        torch.Tensor: The mean of the top s% importance values, differentiable with respect to original weights.
    """
    # Concatenate all importance values into a single tensor
    all_importances = torch.cat([importance.view(-1) for importance in importance_dict.values()])

    # Determine the number of top elements to keep
    k = max(
        1, int(s / 100.0 * all_importances.numel())
    )  # Ensure at least one element is considered

    # Sort the importance values in descending order
    top_importances, _ = torch.topk(all_importances, k)

    # Compute the mean of the top s% importance values
    top_mean = top_importances.mean()

    return top_mean


def optimize_last_n_layers(m, x, N, s, M, lr=1e-3, optimizer=None, verbose=False):
    """
    Optimize the last N layers of the model for M steps using the top s% of weight importance as a TTT loss function.

    Args:
        m (torch.nn.Module): The model.
        x (torch.Tensor): The input batch.
        N (int): The number of layers from the end to optimize.
        s (float): The percentage (0-100) of top importance values to consider for the loss.
        M (int): The number of optimization steps.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-3.

    Returns:
        torch.Tensor: The model's predictions for the input batch after optimization.
    """
    # Set the model to training mode
    m.train()

    # Collect parameters of the last N layers to optimize
    layers = list(m.named_parameters())[-2 * N :]
    params_to_optimize = [param for name, param in layers if "weight" in name]

    # Ensure that the parameters of the last N layers have requires_grad=True
    for param in params_to_optimize:
        param.requires_grad = True

    # Set up the optimizer for the last N layers' parameters
    if optimizer is None:
        optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    # Optimization loop for M steps
    for step in range(M):
        # Compute weight importance
        importance_dict = compute_weight_importance(m, x, N)

        # Calculate the top s% mean importance as the loss
        loss = top_s_percent_mean(importance_dict, s)

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass to compute gradients
        loss.backward()

        # Step the optimizer to update parameters
        optimizer.step()

        # Optionally, print loss for monitoring
        if verbose:
            print(f"Step {step+1}/{M}, Loss: {loss.item()}")

    # After optimization, run the model on the input x to get predictions
    m.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        predictions = m(x)

    return predictions, optimizer


def compute_weight_importance(m, x, N):
    """
    Compute the weight importance for the last N layers of the model efficiently.

    Args:
        m (torch.nn.Module): The model.
        x (torch.Tensor): The input batch.
        N (int): The number of layers from the end to consider.

    Returns:
        dict: A dictionary mapping layer names to the weight importance tensor.
    """
    # Ensure the model is in evaluation mode
    m.eval()

    # Disable gradients for all parameters initially
    for param in m.parameters():
        param.requires_grad = False

    # Enable gradients only for the last N layers
    layers = list(m.named_parameters())[-2 * N :]  # Get the last N layers (both weights and biases)
    for name, param in layers:
        if "weight" in name:
            param.requires_grad = True

    # Forward pass to compute the output
    output = m(x)

    # Use a simple loss function (sum of outputs) to create a scalar output
    loss = output.mean()

    # Compute gradients with respect to the last N layers' parameters
    gradients = torch.autograd.grad(
        loss, [param for name, param in layers if "weight" in name], create_graph=True
    )

    # Calculate importance: w * \nabla{w} for the last N layers
    importance_dict = {}
    grad_idx = 0
    for name, param in layers:
        if "weight" in name:
            importance = (param * gradients[grad_idx]).abs()  # w * \nabla{w}
            # importance = (param).abs()  # w * \nabla{w}
            # importance = gradients[grad_idx].abs()  #  # w * \nabla{w}
            importance_dict[name] = importance  # Don't detach the importance
            grad_idx += 1

    return importance_dict
