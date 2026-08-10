import torch
from torch import nn


class NextTokenLoss(nn.Module):
    """
    Cross-entropy loss for autoregressive next-token prediction.
    """

    def __init__(self) -> None:
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:
                [batch, sequence, vocabulary]

            targets:
                [batch, sequence]

        Returns:
            Scalar cross-entropy loss.
        """

        batch_size, sequence_length, vocab_size = logits.shape

        if targets.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                "targets must have shape "
                "[batch, sequence]."
            )

        # CrossEntropyLoss expects:
        # predictions -> [N, C]
        # targets     -> [N]
        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size,
        )

        targets = targets.reshape(
            batch_size * sequence_length,
        )

        return self.loss_fn(
            logits,
            targets,
        )