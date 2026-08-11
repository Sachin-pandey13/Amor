from pathlib import Path

import torch

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.training.checkpoint import load_checkpoint
from src.amor.training.optimizer import create_optimizer
from src.amor.training.scheduler import WarmupCosineScheduler


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "amor_100k_controlled.pt"
)


def main() -> None:
    print("=" * 70)
    print("AMOR CHECKPOINT VALIDATION")
    print("=" * 70)

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
    # 1. Check checkpoint
    # ---------------------------------------------------------

    print("\n[1/4] Checking checkpoint...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    # ---------------------------------------------------------
    # 2. Recreate model
    # ---------------------------------------------------------

    print("\n[2/4] Creating AMOR model...")

    config = AMORConfig(
        vocab_size=32000,
        dim=256,
        num_heads=8,
        num_layers=4,
        ff_hidden_dim=1024,
        max_seq_len=128,
    )

    model = AMORModel(config).to(device)

    # ---------------------------------------------------------
    # 3. Recreate optimizer + scheduler
    # ---------------------------------------------------------

    print(
        "\n[3/4] Creating optimizer "
        "and scheduler..."
    )

    optimizer = create_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=10,
        max_steps=100,
        min_lr=3e-5,
    )

    # ---------------------------------------------------------
    # 4. Load checkpoint
    # ---------------------------------------------------------

    print("\n[4/4] Loading checkpoint...")

    metadata = load_checkpoint(
        path=str(CHECKPOINT_PATH),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    print(
        f"Restored training step: "
        f"{metadata['step']}"
    )

    print(
        f"Restored config: "
        f"{metadata['config']}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if metadata["step"] != 100:
        raise RuntimeError(
            "Checkpoint does not contain "
            "the expected training step."
        )

    model.eval()

    # Dummy token sequence.
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 32),
        device=device,
    )

    with torch.no_grad():
        logits = model(input_ids)

    expected_shape = (
        2,
        32,
        config.vocab_size,
    )

    print(
        f"Logits shape: {tuple(logits.shape)}"
    )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected logits shape: "
            f"{tuple(logits.shape)}"
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Checkpoint produced "
            "non-finite logits."
        )

    print("\n" + "-" * 70)
    print("CHECKPOINT VALIDATION PASSED")
    print("-" * 70)


if __name__ == "__main__":
    main()