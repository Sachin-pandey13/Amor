import pytest
import torch
from torch import nn

from src.amor.training.optimizer import create_optimizer


class SmallModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(
            16,
            32,
        )

        self.norm = nn.LayerNorm(32)


def test_optimizer_creation():
    model = SmallModel()

    optimizer = create_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    assert isinstance(
        optimizer,
        torch.optim.AdamW,
    )


def test_optimizer_has_two_parameter_groups():
    model = SmallModel()

    optimizer = create_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    assert len(
        optimizer.param_groups
    ) == 2


def test_decay_group_has_weight_decay():
    model = SmallModel()

    optimizer = create_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    decay_values = {
        group["weight_decay"]
        for group in optimizer.param_groups
    }

    assert 0.1 in decay_values
    assert 0.0 in decay_values


def test_invalid_learning_rate():
    model = SmallModel()

    with pytest.raises(ValueError):
        create_optimizer(
            model=model,
            learning_rate=0,
            weight_decay=0.1,
        )


def test_invalid_weight_decay():
    model = SmallModel()

    with pytest.raises(ValueError):
        create_optimizer(
            model=model,
            learning_rate=3e-4,
            weight_decay=-0.1,
        )