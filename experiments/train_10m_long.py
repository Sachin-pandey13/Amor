from pathlib import Path
import json

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
    / "amor_1m"
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
    / "amor_1m_long.pt"
)

METRICS_PATH = (
    ROOT
    / "experiments"
    / "runs"
    / "amor_1m_long_metrics.json"
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
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
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
    print(
        "AMOR 1M LONG TRAINING EXPERIMENT"
    )
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

    eval_interval = 100

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
    print(
        f"Batch size:                  {batch_size}"
    )
    print(
        f"Sequence length:             {sequence_length}"
    )
    print(
        f"Learning rate:               {learning_rate}"
    )
    print(
        f"Minimum learning rate:       {min_learning_rate}"
    )
    print(
        f"Weight decay:                {weight_decay}"
    )
    print(
        f"Maximum steps:               {max_steps}"
    )
    print(
        f"Warmup steps:                {warmup_steps}"
    )
    print(
        f"Gradient clip norm:          "
        f"{gradient_clip_norm}"
    )
    print(
        "Gradient accumulation:       "
        f"{gradient_accumulation_steps}"
    )
    print(
        f"AMP enabled:                 {use_amp}"
    )
    print(
        f"Evaluation interval:         "
        f"{eval_interval}"
    )

    # ---------------------------------------------------------
    # 1. Encode corpus
    # ---------------------------------------------------------

    print(
        "\n[1/8] Encoding 500K corpus..."
    )

    if not CORPUS_PATH.exists():

        raise FileNotFoundError(
            f"Corpus does not exist: "
            f"{CORPUS_PATH}"
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

    # ---------------------------------------------------------
    # 2. Dataset
    # ---------------------------------------------------------

    print(
        "\n[2/8] Creating dataset..."
    )

    dataset = TokenSequenceDataset(
        token_ids=token_ids,
        sequence_length=sequence_length,
    )

    print(
        f"Training sequences: "
        f"{len(dataset):,}"
    )

    if len(dataset) < 2:

        raise RuntimeError(
            "Not enough training sequences."
        )

    # ---------------------------------------------------------
    # 3. Train/validation split
    # ---------------------------------------------------------

    print(
        "\n[3/8] Creating train/validation split..."
    )

    validation_size = max(
        1,
        int(len(dataset) * 0.10),
    )

    training_size = (
        len(dataset) - validation_size
    )

    train_dataset, validation_dataset = (
        random_split(
            dataset,
            [
                training_size,
                validation_size,
            ],
            generator=torch.Generator().manual_seed(
                seed
            ),
        )
    )

    print(
        f"Training sequences:   "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation sequences: "
        f"{len(validation_dataset):,}"
    )

    # ---------------------------------------------------------
    # 4. DataLoaders
    # ---------------------------------------------------------

    print(
        "\n[4/8] Creating DataLoaders..."
    )

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

    first_batch = next(
        iter(train_loader)
    )

    print(
        "Training batch shape: "
        f"{tuple(first_batch.shape)}"
    )

    # ---------------------------------------------------------
    # 5. Model
    # ---------------------------------------------------------

    print(
        "\n[5/8] Creating AMOR model..."
    )

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
        f"Parameters: "
        f"{parameter_count:,}"
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
    # 7. Training with periodic validation
    # ---------------------------------------------------------

    print(
        "\n[7/8] Running 500K long training..."
    )
    print("-" * 70)

    results = []
    metrics = []

    best_validation_loss = float(
        "inf"
    )

    best_step = 0

    train_iterator = iter(
        train_loader
    )

    last_recorded_step = -1

    while (
        trainer.step_count
        < max_steps
    ):

        try:

            batch = next(
                train_iterator
            )

        except StopIteration:

            train_iterator = iter(
                train_loader
            )

            batch = next(
                train_iterator
            )

        result = trainer.train_step(
            batch
        )

        current_step = (
            trainer.step_count
        )

        # Only record completed
        # optimizer steps.
        if (
            current_step
            != last_recorded_step
        ):

            last_recorded_step = (
                current_step
            )

            # Ignore initial
            # accumulation state.
            if current_step == 0:
                continue

            results.append(
                result
            )

            # -------------------------------------------------
            # Periodic validation
            # -------------------------------------------------

            if (
                current_step
                % eval_interval
                == 0
            ):

                validation_loss = evaluate(
                    model=model,
                    dataloader=(
                        validation_loader
                    ),
                    device=device,
                )

                metrics.append(
                    {
                        "step": current_step,
                        "train_loss": result.loss,
                        "validation_loss": (
                            validation_loss
                        ),
                        "learning_rate": (
                            result.learning_rate
                        ),
                    }
                )

                print(
                    f"Step {current_step:04d} | "
                    f"Train Loss: "
                    f"{result.loss:.6f} | "
                    f"Validation Loss: "
                    f"{validation_loss:.6f} | "
                    f"LR: "
                    f"{result.learning_rate:.8f}"
                )

                if (
                    validation_loss
                    < best_validation_loss
                ):

                    best_validation_loss = (
                        validation_loss
                    )

                    best_step = (
                        current_step
                    )

                    print(
                        "  -> New best "
                        "validation loss: "
                        f"{best_validation_loss:.6f}"
                    )

    print(
        "\nTraining complete."
    )

    print(
        f"Final training step: "
        f"{trainer.step_count}"
    )

    print(
        f"Final training loss: "
        f"{results[-1].loss:.6f}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Best validation step: "
        f"{best_step}"
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    print(
        "\nSaving training metrics..."
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_payload = {
        "experiment": (
            "AMOR-1M-long"
        ),
        "metrics": metrics,
        "best_validation_loss": (
            best_validation_loss
        ),
        "best_step": best_step,
        "final_training_loss": (
            results[-1].loss
        ),
        "final_step": (
            trainer.step_count
        ),
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_payload,
            file,
            indent=2,
        )

    print(
        f"Metrics: {METRICS_PATH}"
    )

    # ---------------------------------------------------------
    # 8. Final validation
    # ---------------------------------------------------------

    print(
        "\n[8/8] Running final validation..."
    )

    final_validation_loss = evaluate(
        model=model,
        dataloader=validation_loader,
        device=device,
    )

    print(
        "Final validation loss: "
        f"{final_validation_loss:.6f}"
    )

    # ---------------------------------------------------------
    # Save checkpoint
    # ---------------------------------------------------------

    print(
        "\nSaving 500K long-training checkpoint..."
    )

    save_checkpoint(
        path=str(CHECKPOINT_PATH),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=trainer.step_count,
        config={
            "experiment": (
                "AMOR-1M-long"
            ),
            "corpus": (
                "AMOR-1M-corpus"
            ),
            "token_count": len(
                token_ids
            ),
            "sequence_length": (
                sequence_length
            ),
            "batch_size": batch_size,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": (
                learning_rate
            ),
            "min_learning_rate": (
                min_learning_rate
            ),
            "weight_decay": weight_decay,
            "gradient_clip_norm": (
                gradient_clip_norm
            ),
            "gradient_accumulation_steps": (
                gradient_accumulation_steps
            ),
            "use_amp": use_amp,
            "seed": seed,
            "model_parameters": (
                parameter_count
            ),
            "evaluation_interval": (
                eval_interval
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
            "best_validation_step": (
                best_step
            ),
            "final_validation_loss": (
                final_validation_loss
            ),
        },
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    # ---------------------------------------------------------
    # Final validation checks
    # ---------------------------------------------------------

    if (
        trainer.step_count
        != max_steps
    ):

        raise RuntimeError(
            "Training did not complete "
            "the requested number of steps."
        )

    if (
        trainer.step_count
        != max_steps
    ):

        raise RuntimeError(
            "Trainer step count is incorrect."
        )

    if not metrics:

        raise RuntimeError(
            "No validation metrics "
            "were recorded."
        )

    if not torch.isfinite(
        torch.tensor(
            final_validation_loss
        )
    ):

        raise RuntimeError(
            "Validation loss is not finite."
        )

    if not CHECKPOINT_PATH.exists():

        raise RuntimeError(
            "Checkpoint was not created."
        )

    if not METRICS_PATH.exists():

        raise RuntimeError(
            "Metrics file was not created."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "AMOR 1M LONG TRAINING PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()