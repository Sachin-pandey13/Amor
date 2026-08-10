import pytest
import torch

from src.amor.training.scheduler import WarmupCosineScheduler


def create_scheduler():
    parameter = torch.nn.Parameter(
        torch.tensor(1.0)
    )

    optimizer = torch.optim.AdamW(
        [parameter],
        lr=3e-4,
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=10,
        max_steps=100,
        min_lr=3e-5,
    )

    return scheduler


def test_warmup_increases_learning_rate():
    scheduler = create_scheduler()

    learning_rates = []

    for _ in range(10):
        scheduler.step()

        learning_rates.append(
            scheduler.get_last_lr()[0]
        )

    assert learning_rates[0] < learning_rates[-1]


def test_learning_rate_reaches_base_rate():
    scheduler = create_scheduler()

    for _ in range(10):
        scheduler.step()

    assert scheduler.get_last_lr()[0] == pytest.approx(
        3e-4,
        rel=1e-5,
    )


def test_cosine_decay():
    scheduler = create_scheduler()

    learning_rates = []

    for _ in range(100):
        scheduler.step()

        learning_rates.append(
            scheduler.get_last_lr()[0]
        )

    assert (
        learning_rates[20]
        > learning_rates[50]
        > learning_rates[99]
    )


def test_final_learning_rate():
    scheduler = create_scheduler()

    for _ in range(100):
        scheduler.step()

    assert scheduler.get_last_lr()[0] == pytest.approx(
        3e-5,
        rel=1e-5,
    )


def test_invalid_warmup():
    parameter = torch.nn.Parameter(
        torch.tensor(1.0)
    )

    optimizer = torch.optim.AdamW(
        [parameter],
        lr=3e-4,
    )

    with pytest.raises(ValueError):
        WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=100,
            max_steps=100,
            min_lr=3e-5,
        )