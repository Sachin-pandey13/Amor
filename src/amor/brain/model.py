import torch
from torch import nn

from .config import AMORConfig
from .normalization import RMSNorm
from .transformer_block import TransformerBlock


class AMORModel(nn.Module):
    """
    AMOR-B0 decoder-only Transformer.

    Architecture:

        Token IDs
            ↓
        Token Embedding
            ↓
        Transformer Blocks × N
            ↓
        Final RMSNorm
            ↓
        Language Model Head
            ↓
        Logits
    """

    def __init__(
        self,
        config: AMORConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.dim,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.dim,
                    num_heads=config.num_heads,
                    ff_hidden_dim=config.ff_hidden_dim,
                    max_seq_len=config.max_seq_len,
                    rope_base=config.rope_base,
                    norm_eps=config.norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            dim=config.dim,
            eps=config.norm_eps,
        )

        # Weight tying:
        # LM head uses the same matrix as token embeddings.
        self.lm_head = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = (
            self.token_embedding.weight
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:
                [batch, sequence]

        Returns:
            logits:
                [batch, sequence, vocab_size]
        """

        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits