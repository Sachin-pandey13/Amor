from collections import Counter
import json
from pathlib import Path


def audit_jsonl(
    input_path: str,
    min_tokens: int = 10,
) -> dict:
    """
    Audit a processed JSONL corpus.

    The audit does not modify the corpus.

    It reports:

    - document counts
    - token statistics
    - character statistics
    - source distribution
    - empty documents
    - short documents
    - duplicate hashes
    - missing metadata
    """

    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(
            "Input corpus does not exist: "
            f"{input_file}"
        )

    if min_tokens < 0:
        raise ValueError(
            "min_tokens cannot be negative."
        )

    total_records = 0
    total_tokens = 0
    total_characters = 0

    empty_documents = 0
    short_documents = 0
    missing_source = 0

    token_lengths = []

    source_distribution = Counter()
    hash_counts = Counter()

    with input_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON on line "
                    f"{line_number}."
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                continue

            total_records += 1

            text = record.get(
                "text",
                "",
            )

            if not isinstance(
                text,
                str,
            ):
                text = ""

            text = text.strip()

            if not text:
                empty_documents += 1

            total_characters += len(
                text
            )

            token_count = record.get(
                "token_count",
                0,
            )

            if not isinstance(
                token_count,
                int,
            ):
                token_count = 0

            token_lengths.append(
                token_count
            )

            total_tokens += token_count

            if token_count < min_tokens:
                short_documents += 1

            source = record.get(
                "source"
            )

            if isinstance(
                source,
                str,
            ) and source.strip():
                source_distribution[
                    source
                ] += 1
            else:
                missing_source += 1

            text_hash = record.get(
                "text_hash"
            )

            if isinstance(
                text_hash,
                str,
            ) and text_hash:
                hash_counts[
                    text_hash
                ] += 1

    duplicate_hashes = sum(
        1
        for count in hash_counts.values()
        if count > 1
    )

    if token_lengths:
        min_tokens_value = min(
            token_lengths
        )

        max_tokens_value = max(
            token_lengths
        )

        average_tokens = (
            total_tokens
            / len(token_lengths)
        )
    else:
        min_tokens_value = 0
        max_tokens_value = 0
        average_tokens = 0.0

    return {
        "total_records": total_records,
        "total_tokens": total_tokens,
        "total_characters": total_characters,
        "average_tokens": average_tokens,
        "min_tokens": min_tokens_value,
        "max_tokens": max_tokens_value,
        "empty_documents": empty_documents,
        "short_documents": short_documents,
        "duplicate_hashes": duplicate_hashes,
        "missing_source": missing_source,
        "source_distribution": dict(
            source_distribution
        ),
    }