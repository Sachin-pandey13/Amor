import torch
from torch.utils.data import DataLoader

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.training.dataset import TokenSequenceDataset
from src.amor.training.optimizer import create_optimizer
from src.amor.training.scheduler import WarmupCosineScheduler
from src.amor.training.trainer import Trainer


def create_trainer(
    gradient_accumulation_steps=1,
    use_amp=False,
):
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

    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        use_amp=use_amp,
    )


def test_gradient_accumulation_delays_optimizer_step():
    trainer = create_trainer(
        gradient_accumulation_steps=2,
    )

    batch = torch.randint(
        0,
        128,
        (2, 16),
    )

    result = trainer.train_step(batch)

    assert trainer.step_count == 0
    assert result.step == 0

    result = trainer.train_step(batch)

    assert trainer.step_count == 1
    assert result.step == 1


def test_gradient_accumulation_three_steps():
    trainer = create_trainer(
        gradient_accumulation_steps=3,
    )

    batch = torch.randint(
        0,
        128,
        (2, 16),
    )

    trainer.train_step(batch)

    assert trainer.step_count == 0

    trainer.train_step(batch)

    assert trainer.step_count == 0

    trainer.train_step(batch)

    assert trainer.step_count == 1


def test_amp_disabled_on_cpu():
    trainer = create_trainer(
        use_amp=True,
    )

    assert trainer.use_amp is False


def test_invalid_gradient_accumulation():
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

    import pytest

    with pytest.raises(ValueError):
        Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device("cpu"),
            gradient_accumulation_steps=0,
        )


def test_train_returns_optimizer_steps():
    trainer = create_trainer(
        gradient_accumulation_steps=2,
    )

    token_ids = list(range(128)) * 4

    dataset = TokenSequenceDataset(
        token_ids=token_ids,
        sequence_length=16,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    results = trainer.train(
        dataloader=dataloader,
        max_steps=3,
    )

    assert len(results) == 3
    assert trainer.step_count == 3