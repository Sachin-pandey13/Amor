from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.training.checkpoint import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "amor_100k_controlled.pt"
)

TOKENIZER_PATH = (
    ROOT
    / "data"
    / "tokenizer"
    / "amor_tokenizer.json"
)


def main() -> None:
    print("=" * 70)
    print("AMOR GENERATION DIAGNOSTICS")
    print("=" * 70)

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
    # 2. Load tokenizer
    # ---------------------------------------------------------

    print("\n[1/5] Loading tokenizer...")

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    vocab_size = tokenizer.get_vocab_size()

    print(
        f"Vocabulary size: {vocab_size:,}"
    )

    # ---------------------------------------------------------
    # 3. Create model
    # ---------------------------------------------------------

    print("\n[2/5] Creating model...")

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
    # 4. Load checkpoint
    # ---------------------------------------------------------

    print("\n[3/5] Loading checkpoint...")

    metadata = load_checkpoint(
        path=str(CHECKPOINT_PATH),
        model=model,
        device=device,
    )

    print(
        f"Checkpoint step: "
        f"{metadata['step']}"
    )

    model.eval()

    # ---------------------------------------------------------
    # 5. Inspect predictions
    # ---------------------------------------------------------

    prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "In the modern world",
    ]

    print("\n[4/5] Inspecting next-token predictions...")
    print("-" * 70)

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        encoded = tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long,
            device=device,
        )

        print(
            f"Token count: "
            f"{len(encoded.ids)}"
        )

        print(
            f"Token IDs: "
            f"{encoded.ids}"
        )

        print(
            f"Tokens: "
            f"{encoded.tokens}"
        )

        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[
            0,
            -1,
        ]

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        top_probabilities, top_ids = torch.topk(
            probabilities,
            k=10,
        )

        print("\nTop 10 next-token predictions:")

        for rank, (
            token_id,
            probability,
        ) in enumerate(
            zip(
                top_ids.tolist(),
                top_probabilities.tolist(),
            ),
            start=1,
        ):
            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
            )

            print(
                f"{rank:02d}. "
                f"ID={token_id:<6} "
                f"Prob={probability:.6f} "
                f"Token={token_text!r}"
            )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print("\n[5/5] Validating logits...")

    for prompt in prompts:
        encoded = tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            logits = model(input_ids)

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "Non-finite logits detected."
            )

        if logits.shape[-1] != vocab_size:
            raise RuntimeError(
                "Logit vocabulary size does not "
                "match tokenizer vocabulary size."
            )

    print("\n" + "-" * 70)
    print("GENERATION DIAGNOSTICS PASSED")
    print("-" * 70)
    print("=" * 70)


if __name__ == "__main__":
    main()