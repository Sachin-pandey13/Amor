import torch
from torch.utils.data import DataLoader

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.training.dataset import TokenSequenceDataset
from src.amor.training.optimizer import create_optimizer
from src.amor.training.scheduler import WarmupCosineScheduler
from src.amor.training.trainer import Trainer


def main() -> None:
    torch.manual_seed(42)

    # Small model for the sanity check.
    config = AMORConfig(
        vocab_size=128,
        dim=64,
        num_heads=4,
        num_layers=2,
        ff_hidden_dim=256,
        max_seq_len=32,
    )

    model = AMORModel(config)

    optimizer = create_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.0,
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=10,
        max_steps=500,
        min_lr=3e-5,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    )

    # --------------------------------
    # Tiny deterministic dataset
    # --------------------------------

    sequence = [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        10,
        20,
        30,
        40,
    ]

    token_ids = sequence * 16

    dataset = TokenSequenceDataset(
        token_ids=token_ids,
        sequence_length=16,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    print("AMOR Tiny-Batch Overfit")
    print("=======================")
    print(
        f"Device: {trainer.device}"
    )

    initial_loss = None
    final_loss = None

    for step in range(500):
        # Cycle through the tiny dataset.
        batch = next(
            iter(dataloader)
        )

        result = trainer.train_step(
            batch
        )

        if initial_loss is None:
            initial_loss = result.loss

        final_loss = result.loss

        if (
            result.step == 1
            or result.step % 25 == 0
        ):
            print(
                f"Step {result.step:4d} | "
                f"Loss {result.loss:.6f} | "
                f"LR {result.learning_rate:.8f}"
            )

    print()
    print(
        f"Initial loss: {initial_loss:.6f}"
    )

    print(
        f"Final loss:   {final_loss:.6f}"
    )

    print(
        f"Loss reduction: "
        f"{initial_loss / final_loss:.2f}x"
    )


if __name__ == "__main__":
    main()