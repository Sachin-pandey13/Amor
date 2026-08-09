import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    Applies position-dependent rotations to the final
    dimension of query/key tensors.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(
                "RoPE dimension must be even."
            )

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )

        frequencies = torch.outer(
            positions,
            inv_freq,
        )

        self.register_buffer(
            "cos_cached",
            frequencies.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            frequencies.sin(),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Tensor shaped [batch, heads, seq_len, dim].

            position_offset:
                Starting position for cached decoding.

        Returns:
            Tensor with RoPE applied.
        """

        seq_len = x.size(-2)

        if position_offset + seq_len > self.max_seq_len:
            raise ValueError(
                "Sequence exceeds configured "
                "maximum length."
            )

        cos = self.cos_cached[
            position_offset:
            position_offset + seq_len
        ]

        sin = self.sin_cached[
            position_offset:
            position_offset + seq_len
        ]

        # [seq_len, dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        output = torch.empty_like(x)

        output[..., 0::2] = rotated_even
        output[..., 1::2] = rotated_odd

        return output