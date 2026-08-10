import torch

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from src.amor.training.optimizer import create_optimizer
from src.amor.training.scheduler import WarmupCosineScheduler
from src.amor.training.trainer import Trainer


def create_training_objects():
    config = AMORConfig(
        vocab_size=128,
        dim=32,
        num_heads=4,
        num_layers=2,
        ff_hidden_dim=128,
        max_seq_len=32,
    )

    model = AMORModel(config)

    optimizer = create_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=2,
        max_steps=10,
        min_lr=3e-5,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
    )

    return (
        config,
        model,
        optimizer,
        scheduler,
        trainer,
    )


def test_checkpoint_resume_training(
    tmp_path,
):
    (
        config,
        model,
        optimizer,
        scheduler,
        trainer,
    ) = create_training_objects()

    batch = torch.randint(
        0,
        128,
        (2, 16),
    )

    # ---------------------------------------------------------
    # Train original model
    # ---------------------------------------------------------

    trainer.train_step(batch)
    trainer.train_step(batch)

    assert trainer.step_count == 2

    checkpoint_path = (
        tmp_path
        / "resume_checkpoint.pt"
    )

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=trainer.step_count,
        config={
            "vocab_size": config.vocab_size,
            "sequence_length": 16,
        },
    )

    # ---------------------------------------------------------
    # Capture saved model state
    # ---------------------------------------------------------

    saved_parameters = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    # ---------------------------------------------------------
    # Create completely fresh training objects
    # ---------------------------------------------------------

    (
        new_config,
        new_model,
        new_optimizer,
        new_scheduler,
        new_trainer,
    ) = create_training_objects()

    assert new_trainer.step_count == 0

    # ---------------------------------------------------------
    # Restore checkpoint
    # ---------------------------------------------------------

    metadata = load_checkpoint(
        path=str(checkpoint_path),
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
    )

    new_trainer.step_count = metadata["step"]

    # ---------------------------------------------------------
    # Verify restored step
    # ---------------------------------------------------------

    assert metadata["step"] == 2
    assert new_trainer.step_count == 2

    assert metadata["config"] == {
        "vocab_size": 128,
        "sequence_length": 16,
    }

    # ---------------------------------------------------------
    # Verify model parameters were restored
    # ---------------------------------------------------------

    for name, parameter in (
        new_model.named_parameters()
    ):
        assert torch.equal(
            saved_parameters[name],
            parameter.detach(),
        )

    # ---------------------------------------------------------
    # Resume training
    # ---------------------------------------------------------

    result = new_trainer.train_step(
        batch
    )

    assert result.step == 3
    assert new_trainer.step_count == 3

    assert result.loss > 0

    assert torch.isfinite(
        torch.tensor(result.loss)
    )


def test_checkpoint_resume_preserves_optimizer_state(
    tmp_path,
):
    (
        config,
        model,
        optimizer,
        scheduler,
        trainer,
    ) = create_training_objects()

    batch = torch.randint(
        0,
        128,
        (2, 16),
    )

    trainer.train_step(batch)

    checkpoint_path = (
        tmp_path
        / "optimizer_checkpoint.pt"
    )

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=trainer.step_count,
        config=None,
    )

    (
        _,
        new_model,
        new_optimizer,
        new_scheduler,
        _,
    ) = create_training_objects()

    assert len(
        new_optimizer.state
    ) == 0

    load_checkpoint(
        path=str(checkpoint_path),
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
    )

    assert len(
        new_optimizer.state
    ) > 0


def test_checkpoint_resume_preserves_scheduler_state(
    tmp_path,
):
    (
        config,
        model,
        optimizer,
        scheduler,
        trainer,
    ) = create_training_objects()

    batch = torch.randint(
        0,
        128,
        (2, 16),
    )

    trainer.train_step(batch)
    trainer.train_step(batch)

    checkpoint_path = (
        tmp_path
        / "scheduler_checkpoint.pt"
    )

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=trainer.step_count,
        config=None,
    )

    original_scheduler_state = {
        key: value
        for key, value
        in vars(scheduler).items()
        if key != "optimizer"
    }

    (
        _,
        new_model,
        new_optimizer,
        new_scheduler,
        _,
    ) = create_training_objects()

    load_checkpoint(
        path=str(checkpoint_path),
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
    )

    for key, value in (
        original_scheduler_state.items()
    ):
        assert getattr(
            new_scheduler,
            key,
        ) == value