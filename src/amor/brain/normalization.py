import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes activations using their root mean square
    without subtracting the mean.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.eps = eps

        # Learnable scaling parameter.
        self.weight = nn.Parameter(
            torch.ones(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean square along the final dimension.
        mean_square = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        # Normalize by RMS.
        x = x * torch.rsqrt(
            mean_square + self.eps
        )

        # Apply learnable scale.
        return self.weight * x