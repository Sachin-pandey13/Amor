import json
from pathlib import Path

from tokenizers import Tokenizer

from src.amor.data.acquisition.config import DATASET_SOURCES
from src.amor.data.acquisition.loader import stream_dataset


TARGET_TOKENS = 10_000_000

TOKENIZER_PATH = (
    "data/tokenizer/amor_tokenizer.json"
)

OUTPUT_PATH = (
    "data/raw/amor_10m/corpus.jsonl"
)

MANIFEST_PATH = (
    "data/raw/amor_10m/manifest.json"
)


def allocate_budgets(
    sources,
    total_tokens: int,
) -> dict[str, int]:
    """
    Allocate the global token budget proportionally
    according to each dataset's configured target_tokens.

    Uses largest-remainder allocation so the final
    budgets sum exactly to total_tokens.
    """

    if total_tokens <= 0:
        raise ValueError(
            "total_tokens must be greater than zero."
        )

    configured_total = sum(
        source.target_tokens
        for source in sources
    )

    if configured_total <= 0:
        raise ValueError(
            "Configured dataset token targets "
            "must sum to a positive value."
        )

    raw_allocations = {}

    for source in sources:
        raw_allocations[source.name] = (
            total_tokens
            * source.target_tokens
            / configured_total
        )

    budgets = {
        name: int(value)
        for name, value in raw_allocations.items()
    }

    remainder = (
        total_tokens
        - sum(budgets.values())
    )

    fractions = sorted(
        raw_allocations.items(),
        key=lambda item: (
            item[1] - int(item[1])
        ),
        reverse=True,
    )

    for index in range(remainder):
        name = fractions[index][0]
        budgets[name] += 1

    return budgets


def count_tokens(
    tokenizer: Tokenizer,
    text: str,
) -> int:
    """
    Count tokens using the exact AMOR tokenizer.
    """

    return len(
        tokenizer.encode(text).ids
    )


def acquire_source(
    source,
    tokenizer: Tokenizer,
    target_tokens: int,
) -> tuple[list[dict], int]:
    """
    Acquire documents from one source until the
    source token budget is reached.

    The Hugging Face dataset remains streamed and
    is never loaded completely into memory.
    """

    documents = []
    actual_tokens = 0

    max_documents = max(
        100,
        target_tokens // 20,
    )

    for record in stream_dataset(
        source,
        max_documents=max_documents,
    ):
        text = record["text"]

        token_count = count_tokens(
            tokenizer,
            text,
        )

        if token_count <= 0:
            continue

        if (
            actual_tokens + token_count
            > target_tokens
        ):
            continue

        record["token_count"] = token_count

        documents.append(record)
        actual_tokens += token_count

        if actual_tokens >= target_tokens:
            break

    return documents, actual_tokens


def main() -> None:

    tokenizer_path = Path(
        TOKENIZER_PATH
    )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            "AMOR tokenizer not found at "
            f"{tokenizer_path}. "
            "Build the tokenizer before "
            "running corpus acquisition."
        )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    budgets = allocate_budgets(
        DATASET_SOURCES,
        TARGET_TOKENS,
    )

    output_path = Path(
        OUTPUT_PATH
    )

    manifest_path = Path(
        MANIFEST_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_documents = []
    source_stats = []

    print(
        "AMOR 10M Corpus Acquisition"
    )

    print(
        "=========================="
    )

    print(
        f"Target tokens: "
        f"{TARGET_TOKENS:,}"
    )

    print()

    for source in DATASET_SOURCES:

        target = budgets[source.name]

        print(
            f"Acquiring: {source.name}"
        )

        print(
            f"Target tokens: "
            f"{target:,}"
        )

        documents, actual_tokens = (
            acquire_source(
                source,
                tokenizer,
                target,
            )
        )

        all_documents.extend(
            documents
        )

        source_stats.append(
            {
                "dataset_name": source.name,
                "dataset_id": source.dataset_id,
                "config": source.config,
                "split": source.split,
                "target_tokens": target,
                "documents": len(documents),
                "actual_tokens": actual_tokens,
            }
        )

        print(
            f"Documents: "
            f"{len(documents)}"
        )

        print(
            f"Tokens: "
            f"{actual_tokens:,}"
        )

        print()

    total_tokens = sum(
        record["token_count"]
        for record in all_documents
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in all_documents:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "corpus_name": (
            "AMOR-10M-corpus"
        ),
        "target_tokens": TARGET_TOKENS,
        "actual_tokens": total_tokens,
        "documents": len(
            all_documents
        ),
        "tokenizer": {
            "path": TOKENIZER_PATH,
            "vocab_size": (
                tokenizer.get_vocab_size()
            ),
        },
        "sources": source_stats,
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "# AMOR 1M Corpus"
    )

    print()

    print(
        f"Target tokens:  "
        f"{TARGET_TOKENS:,}"
    )

    print(
        f"Actual tokens:  "
        f"{total_tokens:,}"
    )

    print(
        f"Documents:      "
        f"{len(all_documents)}"
    )

    print(
        f"Output:         "
        f"{output_path}"
    )

    print(
        f"Manifest:       "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()