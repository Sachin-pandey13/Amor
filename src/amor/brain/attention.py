import math

import torch
from torch import nn

from .rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible by num_heads."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    max_seq_len,
                    max_seq_len,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )

    def _split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def _merge_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape

        x = x.transpose(1, 2)

        return x.contiguous().view(
            batch_size,
            seq_len,
            self.dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        q = self.rope(q)
        k = self.rope(k)

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        attention_scores = (
            attention_scores
            / math.sqrt(self.head_dim)
        )

        mask = self.causal_mask[
            :seq_len,
            :seq_len,
        ]

        attention_scores = attention_scores.masked_fill(
            mask,
            torch.finfo(
                attention_scores.dtype
            ).min,
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        output = torch.matmul(
            attention_weights,
            v,
        )

        output = self._merge_heads(output)

        output = self.out_proj(output)

        if return_attention:
            return output, attention_weights

        return output