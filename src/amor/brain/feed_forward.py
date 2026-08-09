import torch
from torch import nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used inside
    an AMOR Transformer block.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.up_proj = nn.Linear(
            dim,
            hidden_dim,
        )

        self.activation = nn.GELU()

        self.down_proj = nn.Linear(
            hidden_dim,
            dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)

        return x