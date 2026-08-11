from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.amor.brain.config import AMORConfig
from src.amor.brain.model import AMORModel
from src.amor.training.checkpoint import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    ROOT / "checkpoints" / "amor_100k_controlled.pt"
)

TOKENIZER_PATH = (
    ROOT / "data" / "tokenizer" / "amor_tokenizer.json"
)


def top_k_top_p_filter(
    logits: torch.Tensor,
    top_k: int = 50,
    top_p: float = 0.95,
) -> torch.Tensor:
    """Apply top-k and nucleus(top-p) filtering."""

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))

        threshold = torch.topk(
            logits,
            top_k,
        ).values[..., -1, None]

        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )

        sorted_probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )

        remove = (
            cumulative_probabilities > top_p
        )

        # Always keep the first token above
        # the nucleus threshold.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        filtered_sorted_logits = sorted_logits.masked_fill(
            remove,
            float("-inf"),
        )

        logits = torch.full_like(
            logits,
            float("-inf"),
        )

        logits.scatter_(
            -1,
            sorted_indices,
            filtered_sorted_logits,
        )

    return logits


def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    """Generate text using temperature + top-k/top-p sampling."""

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than zero."
        )

    if not 0.0 < top_p <= 1.0:
        raise ValueError(
            "top_p must be in the range (0, 1]."
        )

    model.eval()

    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids[:, -128:]

            logits = model(context)

            next_token_logits = logits[:, -1, :]

            # Temperature controls how sharp/flat
            # the probability distribution is.
            next_token_logits = (
                next_token_logits / temperature
            )

            filtered_logits = top_k_top_p_filter(
                next_token_logits,
                top_k=top_k,
                top_p=top_p,
            )

            probabilities = torch.softmax(
                filtered_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

    return tokenizer.decode(
        input_ids[0].tolist(),
        skip_special_tokens=True,
    )


def main() -> None:
    print("=" * 70)
    print("AMOR SAMPLING GENERATION TEST")
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

    print("\n[1/4] Loading tokenizer...")

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.get_vocab_size():,}"
    )

    print("\n[2/4] Loading model...")

    config = AMORConfig(
        vocab_size=32000,
        dim=256,
        num_heads=8,
        num_layers=4,
        ff_hidden_dim=1024,
        max_seq_len=128,
    )

    model = AMORModel(config).to(device)

    print("\n[3/4] Loading checkpoint...")

    metadata = load_checkpoint(
        path=str(CHECKPOINT_PATH),
        model=model,
        device=device,
    )

    print(
        f"Checkpoint step: "
        f"{metadata['step']}"
    )

    print("\n[4/4] Running sampling...")
    print("-" * 70)

    prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "In the modern world",
    ]

    # Fixed seed makes this first comparison reproducible.
    torch.manual_seed(42)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )

        print(f"Output: {output}")

        if not output.strip():
            raise RuntimeError(
                "Sampling generated empty text."
            )

    print("\n" + "-" * 70)
    print("SAMPLING GENERATION PASSED")
    print("-" * 70)
    print("=" * 70)


if __name__ == "__main__":
    main()