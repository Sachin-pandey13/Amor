import torch
from torch import nn

from src.amor.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from src.amor.training.scheduler import (
    WarmupCosineScheduler,
)


def create_training_objects():
    model = nn.Linear(4, 8)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=2,
        max_steps=10,
        min_lr=3e-5,
    )

    return model, optimizer, scheduler


def test_save_checkpoint(tmp_path):
    model, optimizer, scheduler = (
        create_training_objects()
    )

    path = (
        tmp_path
        / "checkpoint.pt"
    )

    save_checkpoint(
        path=str(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=5,
        config={
            "sequence_length": 32,
        },
    )

    assert path.exists()


def test_load_checkpoint_restores_step_and_config(
    tmp_path,
):
    model, optimizer, scheduler = (
        create_training_objects()
    )

    path = (
        tmp_path
        / "checkpoint.pt"
    )

    config = {
        "sequence_length": 32,
        "batch_size": 2,
    }

    save_checkpoint(
        path=str(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=7,
        config=config,
    )

    result = load_checkpoint(
        path=str(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert result["step"] == 7
    assert result["config"] == config


def test_load_checkpoint_restores_model(
    tmp_path,
):
    model, optimizer, scheduler = (
        create_training_objects()
    )

    path = (
        tmp_path
        / "checkpoint.pt"
    )

    original_weight = (
        model.weight.detach().clone()
    )

    save_checkpoint(
        path=str(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=3,
        config=None,
    )

    with torch.no_grad():
        model.weight.add_(10.0)

    assert not torch.equal(
        original_weight,
        model.weight,
    )

    load_checkpoint(
        path=str(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert torch.equal(
        original_weight,
        model.weight,
    )


def test_load_checkpoint_without_optimizer(
    tmp_path,
):
    model, optimizer, scheduler = (
        create_training_objects()
    )

    path = (
        tmp_path
        / "checkpoint.pt"
    )

    save_checkpoint(
        path=str(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=2,
        config=None,
    )

    result = load_checkpoint(
        path=str(path),
        model=model,
    )

    assert result["step"] == 2


def test_missing_checkpoint_rejected(
    tmp_path,
):
    model, _, _ = (
        create_training_objects()
    )

    missing_path = (
        tmp_path
        / "missing.pt"
    )

    try:
        load_checkpoint(
            path=str(missing_path),
            model=model,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Missing checkpoint should raise "
            "FileNotFoundError."
        )


def test_invalid_checkpoint_rejected(
    tmp_path,
):
    model, _, _ = (
        create_training_objects()
    )

    path = (
        tmp_path
        / "invalid.pt"
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            )
        },
        path,
    )

    try:
        load_checkpoint(
            path=str(path),
            model=model,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid checkpoint should raise "
            "ValueError."
        )