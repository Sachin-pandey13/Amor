from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.data.training_dataset import (
    load_jsonl_documents,
    encode_documents,
)
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
    / "amor_10m"
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
    / "amor_10m_long.pt"
)


METRICS_PATH = (
    ROOT
    / "experiments"
    / "runs"
    / "amor_10m_long_metrics.json"
)


def set_seed(
    seed: int,
) -> None:

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


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

            batch = batch.to(
                device
            )

            input_ids = batch[:, :-1]

            targets = batch[:, 1:]

            logits = model(
                input_ids
            )

            loss = loss_fn(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(
                    -1
                ),
            )

            total_loss += loss.item()

            total_batches += 1

    if total_batches == 0:

        raise RuntimeError(
            "Evaluation dataloader is empty."
        )

    return (
        total_loss
        / total_batches
    )


def main() -> None:

    print("=" * 70)

    print(
        "AMOR 10M LONG TRAINING EXPERIMENT"
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

    max_steps = 10_000

    warmup_steps = 100

    gradient_clip_norm = 1.0

    gradient_accumulation_steps = 4

    use_amp = True

    eval_interval = 500

    validation_fraction = 0.10

    set_seed(
        seed
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

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

    print(
        "\nTraining configuration:"
    )

    print(
        f"Batch size:                  "
        f"{batch_size}"
    )

    print(
        f"Sequence length:             "
        f"{sequence_length}"
    )

    print(
        f"Learning rate:               "
        f"{learning_rate}"
    )

    print(
        f"Minimum learning rate:       "
        f"{min_learning_rate}"
    )

    print(
        f"Weight decay:                "
        f"{weight_decay}"
    )

    print(
        f"Maximum steps:               "
        f"{max_steps}"
    )

    print(
        f"Warmup steps:                "
        f"{warmup_steps}"
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
        f"AMP enabled:                 "
        f"{use_amp}"
    )

    print(
        f"Evaluation interval:         "
        f"{eval_interval}"
    )

    print(
        f"Validation fraction:         "
        f"{validation_fraction}"
    )

    # ---------------------------------------------------------
    # 1. Load documents
    # ---------------------------------------------------------

    print(
        "\n[1/8] Loading corpus documents..."
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

    documents = load_jsonl_documents(
        str(CORPUS_PATH)
    )

    print(
        f"Documents: "
        f"{len(documents):,}"
    )

    if len(documents) < 2:

        raise RuntimeError(
            "Not enough documents for "
            "train/validation split."
        )

    # ---------------------------------------------------------
    # 2. Document-level train/validation split
    # ---------------------------------------------------------

    print(
        "\n[2/8] Creating document-level "
        "train/validation split..."
    )

    generator = torch.Generator().manual_seed(
        seed
    )

    indices = torch.randperm(
        len(documents),
        generator=generator,
    ).tolist()

    validation_size = max(
        1,
        int(
            len(documents)
            * validation_fraction
        ),
    )

    training_indices = indices[
        validation_size:
    ]

    validation_indices = indices[
        :validation_size
    ]

    train_documents = [
        documents[index]
        for index in training_indices
    ]

    validation_documents = [
        documents[index]
        for index in validation_indices
    ]

    print(
        f"Training documents:   "
        f"{len(train_documents):,}"
    )

    print(
        f"Validation documents: "
        f"{len(validation_documents):,}"
    )

    # ---------------------------------------------------------
    # 3. Encode train and validation separately
    # ---------------------------------------------------------

    print(
        "\n[3/8] Encoding train/validation "
        "documents..."
    )

    training_token_ids = encode_documents(
        train_documents,
        str(TOKENIZER_PATH),
    )

    validation_token_ids = encode_documents(
        validation_documents,
        str(TOKENIZER_PATH),
    )

    print(
        f"Training tokens:   "
        f"{len(training_token_ids):,}"
    )

    print(
        f"Validation tokens: "
        f"{len(validation_token_ids):,}"
    )

    if len(training_token_ids) < sequence_length:

        raise RuntimeError(
            "Training corpus is too short "
            "for the requested sequence length."
        )

    if len(validation_token_ids) < sequence_length:

        raise RuntimeError(
            "Validation corpus is too short "
            "for the requested sequence length."
        )

    # ---------------------------------------------------------
    # 4. Create datasets
    # ---------------------------------------------------------

    print(
        "\n[4/8] Creating datasets..."
    )

    train_dataset = TokenSequenceDataset(
        token_ids=training_token_ids,
        sequence_length=sequence_length,
    )

    validation_dataset = TokenSequenceDataset(
        token_ids=validation_token_ids,
        sequence_length=sequence_length,
    )

    print(
        f"Training sequences:   "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation sequences: "
        f"{len(validation_dataset):,}"
    )

    if len(train_dataset) < 2:

        raise RuntimeError(
            "Not enough training sequences."
        )

    if len(validation_dataset) < 1:

        raise RuntimeError(
            "Not enough validation sequences."
        )

    # ---------------------------------------------------------
    # 5. DataLoaders
    # ---------------------------------------------------------

    print(
        "\n[5/8] Creating DataLoaders..."
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
    # 6. Model
    # ---------------------------------------------------------

    print(
        "\n[6/8] Creating AMOR model..."
    )

    config = AMORConfig(
        vocab_size=32000,
        dim=256,
        num_heads=8,
        num_layers=4,
        ff_hidden_dim=1024,
        max_seq_len=sequence_length,
    )

    model = AMORModel(
        config
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
    # 7. Optimizer / scheduler / trainer
    # ---------------------------------------------------------

    print(
        "\n[7/8] Creating optimizer, "
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
    # Training
    # ---------------------------------------------------------

    print(
        "\nRunning corrected 10M long training..."
    )

    print(
        "-" * 70
    )

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
                    dataloader=validation_loader,
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
                    f"Step {current_step:05d} | "
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

    # ---------------------------------------------------------
    # Training complete
    # ---------------------------------------------------------

    print(
        "\nTraining complete."
    )

    print(
        f"Final training step: "
        f"{trainer.step_count}"
    )

    if not results:

        raise RuntimeError(
            "No training results were recorded."
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
            "AMOR-10M-long-document-split"
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
        "training_documents": (
            len(train_documents)
        ),
        "validation_documents": (
            len(validation_documents)
        ),
        "training_tokens": (
            len(training_token_ids)
        ),
        "validation_tokens": (
            len(validation_token_ids)
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
        f"Metrics: "
        f"{METRICS_PATH}"
    )

    # ---------------------------------------------------------
    # Final validation
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
        "\nSaving corrected 10M checkpoint..."
    )

    save_checkpoint(
        path=str(CHECKPOINT_PATH),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=trainer.step_count,
        config={
            "experiment": (
                "AMOR-10M-long-document-split"
            ),
            "corpus": (
                "AMOR-10M-corpus"
            ),
            "token_count": (
                len(training_token_ids)
                + len(validation_token_ids)
            ),
            "training_token_count": (
                len(training_token_ids)
            ),
            "validation_token_count": (
                len(validation_token_ids)
            ),
            "training_documents": (
                len(train_documents)
            ),
            "validation_documents": (
                len(validation_documents)
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
            "validation_fraction": (
                validation_fraction
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
    # Final checks
    # ---------------------------------------------------------

    if (
        trainer.step_count
        != max_steps
    ):

        raise RuntimeError(
            "Training did not complete "
            "the requested number of steps."
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
        "AMOR 10M LONG TRAINING PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()