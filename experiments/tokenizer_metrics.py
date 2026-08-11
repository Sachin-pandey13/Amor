from pathlib import Path

from tokenizers import Tokenizer


TOKENIZER_PATH = Path(
    "data/tokenizer/amor_tokenizer.json"
)


def evaluate_tokenizer(
    tokenizer: Tokenizer,
    name: str,
    text: str,
) -> None:

    encoding = tokenizer.encode(text)

    token_count = len(
        encoding.ids
    )

    character_count = len(text)

    compression = (
        character_count / token_count
        if token_count > 0
        else 0.0
    )

    unknown_count = encoding.tokens.count(
        "<unk>"
    )

    print()
    print(name)
    print("-" * len(name))
    print(
        f"Characters : {character_count}"
    )
    print(
        f"Tokens     : {token_count}"
    )
    print(
        f"Chars/token: {compression:.2f}"
    )
    print(
        f"<unk>      : {unknown_count}"
    )
    print(
        f"Tokens     : {encoding.tokens}"
    )


def main() -> None:

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: "
            f"{TOKENIZER_PATH}"
        )

    print(
        "AMOR PRODUCTION TOKENIZER METRICS"
    )
    print("=" * 70)

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    print(
        f"Tokenizer: {TOKENIZER_PATH}"
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.get_vocab_size():,}"
    )

    samples = {
        "English": (
            "What is quantum computing?"
        ),
        "Code": (
            "def calculate(x): return x * 2"
        ),
        "Math": (
            "2 + 2 = 4"
        ),
        "JSON": (
            '{"tool":"youtube","action":"play"}'
        ),
        "Path": (
            r"D:\Amor\src\amor\brain"
        ),
        "Unicode": (
            "Hello 世界 🚀 नमस्ते"
        ),
        "AI": (
            "The future of artificial "
            "intelligence depends on "
            "machine learning."
        ),
        "Long text": (
            "Machine learning is a field "
            "of artificial intelligence "
            "that enables systems to learn "
            "patterns from data."
        ),
    }

    for name, text in samples.items():
        evaluate_tokenizer(
            tokenizer,
            name,
            text,
        )

    print()
    print(
        "TOKENIZER METRICS PASSED"
    )


if __name__ == "__main__":
    main()