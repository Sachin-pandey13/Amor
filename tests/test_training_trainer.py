import torch
from torch.utils.data import DataLoader

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.training.dataset import TokenSequenceDataset
from src.amor.training.optimizer import create_optimizer
from src.amor.training.scheduler import WarmupCosineScheduler
from src.amor.training.trainer import Trainer


def create_trainer() -> Trainer:
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
    )


def test_train_step():
    trainer = create_trainer()

    batch = torch.randint(
        0,
        128,
        (2, 16),
    )

    result = trainer.train_step(batch)

    assert result.step == 1
    assert result.loss > 0
    assert torch.isfinite(
        torch.tensor(result.loss)
    )
    assert result.learning_rate > 0


def test_train_step_updates_parameters():
    trainer = create_trainer()

    batch = torch.randint(
        0,
        128,
        (2, 16),
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in trainer.model.named_parameters()
    }

    trainer.train_step(batch)

    changed = False

    for name, parameter in trainer.model.named_parameters():
        if not torch.equal(
            before[name],
            parameter.detach(),
        ):
            changed = True
            break

    assert changed


def test_train_multiple_steps():
    trainer = create_trainer()

    batch = torch.randint(
        0,
        128,
        (2, 16),
    )

    losses = []

    for _ in range(3):
        result = trainer.train_step(batch)
        losses.append(result.loss)

    assert trainer.step_count == 3

    assert all(
        torch.isfinite(
            torch.tensor(loss)
        )
        for loss in losses
    )


def test_train_method():
    trainer = create_trainer()

    token_ids = list(
        range(128)
    ) * 4

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
        max_steps=5,
    )

    assert len(results) == 5
    assert trainer.step_count == 5