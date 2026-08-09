import torch
from torch import nn

from .attention import CausalSelfAttention
from .feed_forward import FeedForward
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """
    Single pre-normalized Transformer block.

    Structure:

        x
        │
        ├── RMSNorm
        ├── Causal Self-Attention
        └── Residual
        │
        ├── RMSNorm
        ├── Feed Forward
        └── Residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        max_seq_len: int,
        rope_base: float = 10_000.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(
            dim=dim,
            eps=norm_eps,
        )

        self.attention = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )

        self.ffn_norm = RMSNorm(
            dim=dim,
            eps=norm_eps,
        )

        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=ff_hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Attention sub-layer + residual connection
        x = x + self.attention(
            self.attention_norm(x)
        )

        # Feed-forward sub-layer + residual connection
        x = x + self.feed_forward(
            self.ffn_norm(x)
        )

        return x