from dataclasses import dataclass


@dataclass(frozen=True)
class AMORConfig:
    vocab_size: int = 32_000
    dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ff_hidden_dim: int = 1_024
    max_seq_len: int = 512

    rope_base: float = 10_000.0
    norm_eps: float = 1e-6