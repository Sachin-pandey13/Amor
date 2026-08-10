import json

import pytest

from src.amor.data.training_dataset import (
    encode_jsonl_corpus,
)


TOKENIZER_PATH = (
    "data/tokenizer/amor_tokenizer.json"
)


def test_encode_jsonl_corpus_returns_token_ids(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": "Hello AMOR.",
        },
        {
            "id": "2",
            "text": "Python is useful.",
        },
    ]

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(record)
                + "\n"
            )

    token_ids = encode_jsonl_corpus(
        str(input_path),
        TOKENIZER_PATH,
    )

    assert isinstance(
        token_ids,
        list,
    )

    assert len(token_ids) > 0

    assert all(
        isinstance(token_id, int)
        for token_id in token_ids
    )


def test_encode_jsonl_corpus_is_deterministic(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    record = {
        "id": "1",
        "text": "Hello AMOR.",
    }

    input_path.write_text(
        json.dumps(record)
        + "\n",
        encoding="utf-8",
    )

    first = encode_jsonl_corpus(
        str(input_path),
        TOKENIZER_PATH,
    )

    second = encode_jsonl_corpus(
        str(input_path),
        TOKENIZER_PATH,
    )

    assert first == second


def test_encode_jsonl_corpus_preserves_record_order(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": "First document.",
        },
        {
            "id": "2",
            "text": "Second document.",
        },
    ]

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(record)
                + "\n"
            )

    token_ids = encode_jsonl_corpus(
        str(input_path),
        TOKENIZER_PATH,
    )

    assert len(token_ids) > 0


def test_encode_jsonl_corpus_rejects_missing_input(
    tmp_path,
):
    input_path = (
        tmp_path / "missing.jsonl"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        encode_jsonl_corpus(
            str(input_path),
            TOKENIZER_PATH,
        )


def test_encode_jsonl_corpus_rejects_missing_tokenizer(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    input_path.write_text(
        json.dumps(
            {
                "text": "Hello AMOR.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileNotFoundError
    ):
        encode_jsonl_corpus(
            str(input_path),
            str(
                tmp_path
                / "missing_tokenizer.json"
            ),
        )


def test_encode_jsonl_corpus_rejects_missing_text(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    input_path.write_text(
        json.dumps(
            {
                "id": "bad",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError
    ):
        encode_jsonl_corpus(
            str(input_path),
            TOKENIZER_PATH,
        )


def test_encode_jsonl_corpus_rejects_empty_text(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    input_path.write_text(
        json.dumps(
            {
                "id": "bad",
                "text": "   ",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError
    ):
        encode_jsonl_corpus(
            str(input_path),
            TOKENIZER_PATH,
        )


def test_encode_jsonl_corpus_rejects_invalid_json(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    input_path.write_text(
        '{"text": "valid"}\n'
        '{"broken": \n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON",
    ):
        encode_jsonl_corpus(
            str(input_path),
            TOKENIZER_PATH,
        )