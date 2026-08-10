from pathlib import Path

import torch
from torch.utils.data import Dataset


class TokenSequenceDataset(Dataset):
    """
    Dataset containing pre-tokenized sequences.

    Each item returns one fixed-length sequence of token IDs.
    """

    def __init__(
        self,
        token_ids: list[int],
        sequence_length: int,
    ) -> None:
        if sequence_length < 2:
            raise ValueError(
                "sequence_length must be at least 2."
            )

        if len(token_ids) < sequence_length:
            raise ValueError(
                "Token corpus is shorter than "
                "the requested sequence length."
            )

        self.token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        self.sequence_length = sequence_length

        self.num_sequences = (
            len(self.token_ids)
            // sequence_length
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:
        start = (
            index * self.sequence_length
        )

        end = (
            start + self.sequence_length
        )

        return self.token_ids[
            start:end
        ].clone()