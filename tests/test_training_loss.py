import pytest
import torch

from src.amor.training.loss import NextTokenLoss


def test_loss_returns_scalar():
    loss_fn = NextTokenLoss()

    logits = torch.randn(
        2,
        8,
        100,
    )

    targets = torch.randint(
        0,
        100,
        (2, 8),
    )

    loss = loss_fn(
        logits,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_backward():
    loss_fn = NextTokenLoss()

    logits = torch.randn(
        2,
        8,
        100,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        100,
        (2, 8),
    )

    loss = loss_fn(
        logits,
        targets,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()


def test_loss_rejects_wrong_target_shape():
    loss_fn = NextTokenLoss()

    logits = torch.randn(
        2,
        8,
        100,
    )

    targets = torch.randint(
        0,
        100,
        (2, 7),
    )

    with pytest.raises(ValueError):
        loss_fn(
            logits,
            targets,
        )


def test_perfect_prediction_has_low_loss():
    loss_fn = NextTokenLoss()

    logits = torch.full(
        (1, 3, 10),
        -10.0,
    )

    targets = torch.tensor(
        [[2, 5, 7]]
    )

    logits[0, 0, 2] = 10.0
    logits[0, 1, 5] = 10.0
    logits[0, 2, 7] = 10.0

    loss = loss_fn(
        logits,
        targets,
    )

    assert loss.item() < 0.001