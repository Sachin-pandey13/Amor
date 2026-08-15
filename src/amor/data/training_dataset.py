from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer


def _load_tokenizer(
    tokenizer_path: str,
) -> Tokenizer:
    """
    Load the AMOR tokenizer and verify that the required
    EOS token exists.
    """

    tokenizer_file = Path(tokenizer_path)

    if not tokenizer_file.exists():
        raise FileNotFoundError(
            f"Tokenizer does not exist: {tokenizer_file}"
        )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_file)
    )

    eos_token_id = tokenizer.token_to_id(
        "<eos>"
    )

    if eos_token_id is None:
        raise ValueError(
            "AMOR tokenizer does not contain "
            "the required <eos> token."
        )

    return tokenizer


def load_jsonl_documents(
    input_path: str,
) -> list[str]:
    """
    Load non-empty text documents from a JSONL corpus.

    Each JSONL record must contain a string `text` field.

    The documents remain separate so that the caller can
    perform a document-level train/validation split before
    tokenization.
    """

    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input corpus does not exist: {input_file}"
        )

    documents: list[str] = []

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
                    f"Invalid JSON on line "
                    f"{line_number}."
                ) from exc

            text = record.get("text")

            if not isinstance(text, str):
                raise ValueError(
                    f"Record on line "
                    f"{line_number} must contain "
                    "a string 'text' field."
                )

            text = text.strip()

            if not text:
                raise ValueError(
                    f"Record on line "
                    f"{line_number} contains "
                    "empty text."
                )

            documents.append(text)

    return documents


def encode_documents(
    documents: list[str],
    tokenizer_path: str,
) -> list[int]:
    """
    Encode a list of documents into one flat token sequence.

    An EOS token is inserted after every document.

    Structure:

        document A -> tokens -> <eos>
        document B -> tokens -> <eos>
        document C -> tokens -> <eos>

    This prevents unrelated documents from being treated
    as one continuous piece of text.
    """

    tokenizer = _load_tokenizer(
        tokenizer_path
    )

    eos_token_id = tokenizer.token_to_id(
        "<eos>"
    )

    # `_load_tokenizer` guarantees that this exists.
    assert eos_token_id is not None

    token_ids: list[int] = []

    for document in documents:

        text = document.strip()

        if not text:
            continue

        encoded = tokenizer.encode(
            text
        )

        if not encoded.ids:
            continue

        token_ids.extend(
            encoded.ids
        )

        # Explicit document boundary.
        token_ids.append(
            eos_token_id
        )

    return token_ids


def encode_jsonl_corpus(
    input_path: str,
    tokenizer_path: str,
) -> list[int]:
    """
    Encode an entire JSONL corpus into one token sequence.

    Each document is terminated with the AMOR EOS token.

    This function is retained for compatibility with existing
    experiments that directly encode a complete corpus.

    For the corrected training pipeline, prefer:

        load_jsonl_documents()
        -> document-level split
        -> encode_documents()
    """

    documents = load_jsonl_documents(
        input_path
    )

    return encode_documents(
        documents,
        tokenizer_path,
    )