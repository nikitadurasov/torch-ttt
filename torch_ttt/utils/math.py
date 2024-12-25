import torch


def compute_covariance(features: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Compute covariance matrix for given features along a specific dimension.

    Args:
        features (torch.Tensor): Input tensor of shape [N, D] or higher dimensions.
        dim (int): The dimension along which to compute covariance.

    Returns:
        torch.Tensor: Covariance matrix of shape [D, D].

    Raises:
        ValueError: If the input tensor has fewer than 2 dimensions.
    """
    if features.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions to compute covariance.")

    if features.size(dim) <= 1:
        raise ValueError(
            f"Cannot compute covariance with less than 2 samples along dimension {dim}."
        )

    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)

    return cov
