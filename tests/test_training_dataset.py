import pytest
import torch

from src.amor.training.dataset import TokenSequenceDataset


def test_dataset_length():
    token_ids = list(range(100))

    dataset = TokenSequenceDataset(
        token_ids=token_ids,
        sequence_length=10,
    )

    assert len(dataset) == 10


def test_dataset_item_shape():
    token_ids = list(range(100))

    dataset = TokenSequenceDataset(
        token_ids=token_ids,
        sequence_length=10,
    )

    item = dataset[0]

    assert item.shape == (10,)
    assert item.dtype == torch.long


def test_dataset_item_values():
    token_ids = list(range(100))

    dataset = TokenSequenceDataset(
        token_ids=token_ids,
        sequence_length=10,
    )

    item = dataset[2]

    assert torch.equal(
        item,
        torch.tensor(
            list(range(20, 30)),
            dtype=torch.long,
        ),
    )


def test_dataset_rejects_short_corpus():
    with pytest.raises(ValueError):
        TokenSequenceDataset(
            token_ids=[1, 2, 3],
            sequence_length=10,
        )


def test_dataset_rejects_invalid_sequence_length():
    with pytest.raises(ValueError):
        TokenSequenceDataset(
            token_ids=list(range(100)),
            sequence_length=1,
        )