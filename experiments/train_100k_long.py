from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.data.training_dataset import encode_jsonl_corpus
from src.amor.training.checkpoint import save_checkpoint
from src.amor.training.dataset import TokenSequenceDataset
from src.amor.training.optimizer import create_optimizer
from src.amor.training.scheduler import WarmupCosineScheduler
from src.amor.training.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
    ROOT
    / "data"
    / "raw"
    / "amor_100k"
    / "corpus.jsonl"
)

TOKENIZER_PATH = (
    ROOT
    / "data"
    / "tokenizer"
    / "amor_tokenizer.json"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "amor_100k_long.pt"
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate next-token cross-entropy loss
    without updating model parameters.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(input_ids)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError(
            "Evaluation dataloader is empty."
        )

    return total_loss / total_batches


def main() -> None:
    print("=" * 70)
    print("AMOR 100K LONG TRAINING EXPERIMENT")
    print("=" * 70)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    seed = 42

    batch_size = 4
    sequence_length = 128

    learning_rate = 3e-4
    min_learning_rate = 3e-5
    weight_decay = 0.1

    max_steps = 1_000
    warmup_steps = 100

    gradient_clip_norm = 1.0
    gradient_accumulation_steps = 4
    use_amp = True

    set_seed(seed)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "CUDA memory available: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # ---------------------------------------------------------
    # Configuration summary
    # ---------------------------------------------------------

    print("\nTraining configuration:")
    print(f"Batch size:                  {batch_size}")
    print(f"Sequence length:             {sequence_length}")
    print(f"Learning rate:               {learning_rate}")
    print(f"Minimum learning rate:       {min_learning_rate}")
    print(f"Weight decay:                {weight_decay}")
    print(f"Maximum steps:               {max_steps}")
    print(f"Warmup steps:                {warmup_steps}")
    print(f"Gradient clip norm:          {gradient_clip_norm}")
    print(
        "Gradient accumulation:       "
        f"{gradient_accumulation_steps}"
    )
    print(f"AMP enabled:                 {use_amp}")

    # ---------------------------------------------------------
    # 1. Encode corpus
    # ---------------------------------------------------------

    print("\n[1/8] Encoding 100K corpus...")

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus does not exist: {CORPUS_PATH}"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer does not exist: {TOKENIZER_PATH}"
        )

    token_ids = encode_jsonl_corpus(
        str(CORPUS_PATH),
        str(TOKENIZER_PATH),
    )

    print(
        f"Token IDs: {len(token_ids):,}"
    )

    # ---------------------------------------------------------
    # 2. Dataset
    # ---------------------------------------------------------

    print("\n[2/8] Creating dataset...")

    dataset = TokenSequenceDataset(
        token_ids=token_ids,
        sequence_length=sequence_length,
    )

    print(
        f"Training sequences: {len(dataset):,}"
    )

    if len(dataset) < 2:
        raise RuntimeError(
            "Not enough training sequences."
        )

    # ---------------------------------------------------------
    # 3. Train/validation split
    # ---------------------------------------------------------

    print("\n[3/8] Creating train/validation split...")

    validation_size = max(
        1,
        int(len(dataset) * 0.10),
    )

    training_size = len(dataset) - validation_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(
        f"Training sequences:   {len(train_dataset):,}"
    )

    print(
        f"Validation sequences: {len(validation_dataset):,}"
    )

    # ---------------------------------------------------------
    # 4. DataLoaders
    # ---------------------------------------------------------

    print("\n[4/8] Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    first_batch = next(iter(train_loader))

    print(
        f"Training batch shape: "
        f"{tuple(first_batch.shape)}"
    )

    # ---------------------------------------------------------
    # 5. Model
    # ---------------------------------------------------------

    print("\n[5/8] Creating AMOR model...")

    config = AMORConfig(
        vocab_size=32000,
        dim=256,
        num_heads=8,
        num_layers=4,
        ff_hidden_dim=1024,
        max_seq_len=sequence_length,
    )

    model = AMORModel(config)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    # ---------------------------------------------------------
    # 6. Optimizer / scheduler / trainer
    # ---------------------------------------------------------

    print(
        "\n[6/8] Creating optimizer, "
        "scheduler and trainer..."
    )

    optimizer = create_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=min_learning_rate,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_clip_norm=gradient_clip_norm,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        use_amp=use_amp,
    )

    # ---------------------------------------------------------
    # 7. Training
    # ---------------------------------------------------------

    print("\n[7/8] Running long training...")
    print("-" * 70)

    results = trainer.train(
        dataloader=train_loader,
        max_steps=max_steps,
    )

    print("\nTraining complete.")

    print(
        f"Final training step: "
        f"{trainer.step_count}"
    )

    print(
        f"Final training loss: "
        f"{results[-1].loss:.6f}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\n[8/8] Running validation...")

    validation_loss = evaluate(
        model=model,
        dataloader=validation_loader,
        device=device,
    )

    print(
        f"Validation loss: "
        f"{validation_loss:.6f}"
    )

    # ---------------------------------------------------------
    # Save checkpoint
    # ---------------------------------------------------------

    print("\nSaving long-training checkpoint...")

    save_checkpoint(
        path=str(CHECKPOINT_PATH),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=trainer.step_count,
        config={
            "experiment": "AMOR-100K-long",
            "corpus": "AMOR-100K-corpus",
            "token_count": len(token_ids),
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "gradient_accumulation_steps": (
                gradient_accumulation_steps
            ),
            "use_amp": use_amp,
            "seed": seed,
            "model_parameters": parameter_count,
            "validation_loss": validation_loss,
        },
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if len(results) != max_steps:
        raise RuntimeError(
            "Training did not complete "
            "the requested number of steps."
        )

    if trainer.step_count != max_steps:
        raise RuntimeError(
            "Trainer step count is incorrect."
        )

    if not torch.isfinite(
        torch.tensor(validation_loss)
    ):
        raise RuntimeError(
            "Validation loss is not finite."
        )

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            "Checkpoint was not created."
        )

    print("\n" + "=" * 70)
    print("AMOR 100K LONG TRAINING PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()