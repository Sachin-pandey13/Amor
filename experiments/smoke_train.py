from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.data.training_dataset import encode_jsonl_corpus
from src.amor.training.dataset import TokenSequenceDataset
from src.amor.training.optimizer import create_optimizer
from src.amor.training.scheduler import WarmupCosineScheduler
from src.amor.training.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "smoke_corpus_v2.jsonl"
)

TOKENIZER_PATH = (
    ROOT
    / "data"
    / "tokenizer"
    / "amor_tokenizer.json"
)


def main() -> None:
    print("=" * 60)
    print("AMOR END-TO-END TRAINING SMOKE TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ---------------------------------------------------------
    # 2. Encode processed corpus
    # ---------------------------------------------------------

    print("\n[1/6] Encoding corpus...")

    token_ids = encode_jsonl_corpus(
        str(CORPUS_PATH),
        str(TOKENIZER_PATH),
    )

    print(
        f"Token IDs: {len(token_ids):,}"
    )

    if len(token_ids) < 32:
        raise RuntimeError(
            "Corpus is too small for smoke training."
        )

    # ---------------------------------------------------------
    # 3. Create training dataset
    # ---------------------------------------------------------

    print("\n[2/6] Creating dataset...")

    sequence_length = 32

    dataset = TokenSequenceDataset(
        token_ids=token_ids,
        sequence_length=sequence_length,
    )

    print(
        f"Sequence length: {sequence_length}"
    )
    print(
        f"Training sequences: {len(dataset):,}"
    )

    # ---------------------------------------------------------
    # 4. Create DataLoader
    # ---------------------------------------------------------

    print("\n[3/6] Creating DataLoader...")

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    first_batch = next(
        iter(dataloader)
    )

    print(
        f"Batch shape: {tuple(first_batch.shape)}"
    )

    if first_batch.shape != (
        2,
        sequence_length,
    ):
        raise RuntimeError(
            "Unexpected batch shape."
        )

    # ---------------------------------------------------------
    # 5. Create AMOR model
    # ---------------------------------------------------------

    print("\n[4/6] Creating AMOR model...")

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
    # 6. Optimizer + scheduler + trainer
    # ---------------------------------------------------------

    print(
        "\n[5/6] Creating optimizer, "
        "scheduler and trainer..."
    )

    optimizer = create_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=2,
        max_steps=5,
        min_lr=3e-5,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # ---------------------------------------------------------
    # 7. Real training
    # ---------------------------------------------------------

    print("\n[6/6] Running training...")
    print("-" * 60)

    results = trainer.train(
        dataloader=dataloader,
        max_steps=5,
    )

    for result in results:
        print(
            f"Step {result.step:02d} | "
            f"Loss: {result.loss:.6f} | "
            f"LR: {result.learning_rate:.8f}"
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if len(results) != 5:
        raise RuntimeError(
            "Training did not complete "
            "the requested number of steps."
        )

    if trainer.step_count != 5:
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

    print("-" * 60)
    print("SMOKE TRAINING PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()