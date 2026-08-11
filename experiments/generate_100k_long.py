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
    / "amor_100k_long.pt"
)

TOKENIZER_PATH = (
    ROOT
    / "data"
    / "tokenizer"
    / "amor_tokenizer.json"
)


def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
) -> str:
    """
    Generate text autoregressively from a prompt.
    """

    model.eval()

    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Keep context within the model's maximum
            # sequence length.
            context = input_ids[:, -128:]

            logits = model(context)

            # Logits for the final token position.
            next_token_logits = logits[:, -1, :]

            # Greedy decoding:
            # choose the token with the highest probability.
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token,
                ],
                dim=1,
            )

    generated_ids = input_ids[0].tolist()

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


def main() -> None:
    print("=" * 70)
    print("AMOR 100K TEXT GENERATION TEST")
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
    # 2. Validate files
    # ---------------------------------------------------------

    print("\n[1/5] Checking files...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{CHECKPOINT_PATH}"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer does not exist: "
            f"{TOKENIZER_PATH}"
        )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Tokenizer: {TOKENIZER_PATH}"
    )

    # ---------------------------------------------------------
    # 3. Load tokenizer
    # ---------------------------------------------------------

    print("\n[2/5] Loading tokenizer...")

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    vocabulary_size = (
        tokenizer.get_vocab_size()
    )

    print(
        f"Vocabulary size: "
        f"{vocabulary_size:,}"
    )

    if vocabulary_size != 32000:
        raise RuntimeError(
            "Tokenizer vocabulary size does not "
            f"match the model configuration: "
            f"{vocabulary_size}"
        )

    # ---------------------------------------------------------
    # 4. Load model + checkpoint
    # ---------------------------------------------------------

    print("\n[3/5] Loading AMOR model...")

    config = AMORConfig(
        vocab_size=32000,
        dim=256,
        num_heads=8,
        num_layers=4,
        ff_hidden_dim=1024,
        max_seq_len=128,
    )

    model = AMORModel(config).to(device)

    metadata = load_checkpoint(
        path=str(CHECKPOINT_PATH),
        model=model,
        device=device,
    )

    print(
        f"Checkpoint step: "
        f"{metadata['step']}"
    )

    if metadata["step"] != 100:
        raise RuntimeError(
            "Expected checkpoint at training step 100."
        )

    # ---------------------------------------------------------
    # 5. Generate text
    # ---------------------------------------------------------

    print("\n[4/5] Running generation...")
    print("-" * 70)

    prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "In the modern world",
    ]

    outputs = []

    for prompt in prompts:
        print(
            f"\nPrompt: {prompt}"
        )

        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=50,
        )

        print(
            f"Output: {output}"
        )

        if not output.strip():
            raise RuntimeError(
                "Model generated empty text."
            )

        outputs.append(output)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print(
        "\n[5/5] Validating generation..."
    )

    for output in outputs:
        if not output.strip():
            raise RuntimeError(
                "Generation produced empty output."
            )

    print("\n" + "-" * 70)
    print("GENERATION TEST PASSED")
    print("-" * 70)
    print("=" * 70)


if __name__ == "__main__":
    main()