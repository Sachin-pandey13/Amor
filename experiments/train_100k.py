from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.data.training_dataset import (
    encode_jsonl_corpus,
)
from src.amor.training.checkpoint import (
    save_checkpoint,
)
from src.amor.training.config import TrainingConfig
from src.amor.training.dataset import (
    TokenSequenceDataset,
)
from src.amor.training.optimizer import (
    create_optimizer,
)
from src.amor.training.scheduler import (
    WarmupCosineScheduler,
)
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
    / "amor_100k_controlled.pt"
)


def main() -> None:
    print("=" * 70)
    print("AMOR 100K CONTROLLED TRAINING RUN")
    print("=" * 70)

    # ---------------------------------------------------------
    # 0. Training configuration
    # ---------------------------------------------------------

    # Override only the values needed for this controlled run.
    training_config = TrainingConfig(
        batch_size=4,
        sequence_length=128,
        max_steps=100,
        warmup_steps=10,
    )

    print("\nTraining configuration:")
    print(
        f"Batch size:                  "
        f"{training_config.batch_size}"
    )
    print(
        f"Sequence length:             "
        f"{training_config.sequence_length}"
    )
    print(
        f"Learning rate:               "
        f"{training_config.learning_rate}"
    )
    print(
        f"Minimum learning rate:       "
        f"{training_config.min_learning_rate}"
    )
    print(
        f"Weight decay:                "
        f"{training_config.weight_decay}"
    )
    print(
        f"Maximum steps:               "
        f"{training_config.max_steps}"
    )
    print(
        f"Warmup steps:                "
        f"{training_config.warmup_steps}"
    )
    print(
        f"Gradient clip norm:          "
        f"{training_config.gradient_clip_norm}"
    )
    print(
        f"Gradient accumulation:       "
        f"{training_config.gradient_accumulation_steps}"
    )
    print(
        f"AMP enabled:                 "
        f"{training_config.use_amp}"
    )

    # ---------------------------------------------------------
    # 1. Reproducibility
    # ---------------------------------------------------------

    torch.manual_seed(
        training_config.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            training_config.seed
        )

    # ---------------------------------------------------------
    # 2. Device
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
            f"CUDA memory available: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # ---------------------------------------------------------
    # 3. Validate and encode corpus
    # ---------------------------------------------------------

    print("\n[1/7] Encoding 100K corpus...")

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus does not exist: {CORPUS_PATH}"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer does not exist: "
            f"{TOKENIZER_PATH}"
        )

    token_ids = encode_jsonl_corpus(
        str(CORPUS_PATH),
        str(TOKENIZER_PATH),
    )

    print(
        f"Token IDs: {len(token_ids):,}"
    )

    if len(token_ids) < 100_000:
        print(
            "Warning: encoded corpus contains "
            "fewer than 100,000 tokens."
        )

    # ---------------------------------------------------------
    # 4. Create training dataset
    # ---------------------------------------------------------

    print("\n[2/7] Creating training dataset...")

    sequence_length = (
        training_config.sequence_length
    )

    dataset = TokenSequenceDataset(
        token_ids=token_ids,
        sequence_length=sequence_length,
    )

    print(
        f"Sequence length: {sequence_length}"
    )

    print(
        f"Training sequences: "
        f"{len(dataset):,}"
    )

    # ---------------------------------------------------------
    # 5. Create DataLoader
    # ---------------------------------------------------------

    print("\n[3/7] Creating DataLoader...")

    batch_size = (
        training_config.batch_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    first_batch = next(
        iter(dataloader)
    )

    print(
        f"Batch shape: "
        f"{tuple(first_batch.shape)}"
    )

    expected_shape = (
        batch_size,
        sequence_length,
    )

    if first_batch.shape != expected_shape:
        raise RuntimeError(
            "Unexpected batch shape: "
            f"{tuple(first_batch.shape)}"
        )

    # ---------------------------------------------------------
    # 6. Create AMOR model
    # ---------------------------------------------------------

    print("\n[4/7] Creating AMOR model...")

    model_config = AMORConfig(
        vocab_size=32000,
        dim=256,
        num_heads=8,
        num_layers=4,
        ff_hidden_dim=1024,
        max_seq_len=sequence_length,
    )

    model = AMORModel(
        model_config
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # ---------------------------------------------------------
    # 7. Optimizer + scheduler + trainer
    # ---------------------------------------------------------

    print(
        "\n[5/7] Creating optimizer, "
        "scheduler and trainer..."
    )

    optimizer = create_optimizer(
        model=model,
        learning_rate=(
            training_config.learning_rate
        ),
        weight_decay=(
            training_config.weight_decay
        ),
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=(
            training_config.warmup_steps
        ),
        max_steps=(
            training_config.max_steps
        ),
        min_lr=(
            training_config.min_learning_rate
        ),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_clip_norm=(
            training_config.gradient_clip_norm
        ),
        gradient_accumulation_steps=(
            training_config.gradient_accumulation_steps
        ),
        use_amp=(
            training_config.use_amp
        ),
    )

    # ---------------------------------------------------------
    # 8. Controlled training
    # ---------------------------------------------------------

    print(
        "\n[6/7] Running controlled training..."
    )
    print("-" * 70)

    results = trainer.train(
        dataloader=dataloader,
        max_steps=(
            training_config.max_steps
        ),
    )

    for result in results:
        print(
            f"Step {result.step:03d} | "
            f"Loss: {result.loss:.6f} | "
            f"LR: {result.learning_rate:.8f}"
        )

    # ---------------------------------------------------------
    # 9. Save checkpoint
    # ---------------------------------------------------------

    print("\n[7/7] Saving checkpoint...")

    save_checkpoint(
        path=str(CHECKPOINT_PATH),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=trainer.step_count,
        config={
            "corpus": "AMOR-100K-corpus",
            "token_count": len(token_ids),
            "sequence_length": (
                training_config.sequence_length
            ),
            "batch_size": (
                training_config.batch_size
            ),
            "max_steps": (
                training_config.max_steps
            ),
            "warmup_steps": (
                training_config.warmup_steps
            ),
            "learning_rate": (
                training_config.learning_rate
            ),
            "min_learning_rate": (
                training_config.min_learning_rate
            ),
            "weight_decay": (
                training_config.weight_decay
            ),
            "gradient_clip_norm": (
                training_config.gradient_clip_norm
            ),
            "gradient_accumulation_steps": (
                training_config.gradient_accumulation_steps
            ),
            "use_amp": (
                training_config.use_amp
            ),
            "seed": (
                training_config.seed
            ),
            "model_parameters": parameter_count,
        },
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if len(results) != (
        training_config.max_steps
    ):
        raise RuntimeError(
            "Training did not complete "
            "the requested number of steps."
        )

    if trainer.step_count != (
        training_config.max_steps
    ):
        raise RuntimeError(
            "Trainer step count is incorrect."
        )

    for result in results:
        if not torch.isfinite(
            torch.tensor(result.loss)
        ):
            raise RuntimeError(
                "Non-finite loss detected."
            )

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            "Checkpoint was not created."
        )

    print("-" * 70)
    print("100K CONTROLLED TRAINING PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()