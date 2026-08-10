from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer


def encode_jsonl_corpus(
    input_path: str,
    tokenizer_path: str,
) -> list[int]:
    """
    Encode a processed JSONL corpus into one flat
    sequence of tokenizer IDs.

    Each JSONL record must contain a string `text` field.

    The function deliberately does not add BOS/EOS tokens.
    Special-token handling will be introduced separately
    once the training sequence contract explicitly requires it.

    Args:
        input_path:
            Path to the processed JSONL corpus.

        tokenizer_path:
            Path to the trained AMOR tokenizer JSON file.

    Returns:
        A flat list of integer token IDs.

    Raises:
        FileNotFoundError:
            If either input file does not exist.

        ValueError:
            If a JSONL record is invalid or does not contain
            a valid text field.
    """

    input_file = Path(input_path)
    tokenizer_file = Path(tokenizer_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input corpus does not exist: {input_file}"
        )

    if not tokenizer_file.exists():
        raise FileNotFoundError(
            f"Tokenizer does not exist: {tokenizer_file}"
        )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_file)
    )

    token_ids: list[int] = []

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
                    f"Invalid JSON on line {line_number}."
                ) from exc

            text = record.get("text")

            if not isinstance(text, str):
                raise ValueError(
                    f"Record on line {line_number} "
                    "must contain a string 'text' field."
                )

            if not text.strip():
                raise ValueError(
                    f"Record on line {line_number} "
                    "contains empty text."
                )

            encoded = tokenizer.encode(text)

            token_ids.extend(
                encoded.ids
            )

    return token_ids