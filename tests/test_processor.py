import json

from src.amor.data.processor import (
    process_jsonl,
)


def test_process_jsonl_normalizes_and_filters(
    tmp_path,
):
    input_path = (
        tmp_path / "input.jsonl"
    )

    output_path = (
        tmp_path / "output.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": (
                "  Hello     AMOR. "
                "This is a valid training document.  "
            ),
            "token_count": 4,
        },
        {
            "id": "2",
            "text": "",
            "token_count": 0,
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

    stats = process_jsonl(
        str(input_path),
        str(output_path),
    )

    assert stats["total_records"] == 2
    assert stats["accepted_records"] == 1
    assert stats["rejected_records"] == 1
    assert stats["duplicate_records"] == 0
    assert stats["unique_records"] == 1

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        output = [
            json.loads(line)
            for line in file
        ]

    assert len(output) == 1

    assert output[0]["text"] == (
        "Hello AMOR. "
        "This is a valid training document."
    )

    assert output[0]["text_hash"]
    assert len(output[0]["text_hash"]) == 64

    assert output[0]["token_count"] > 0


def test_process_jsonl_preserves_metadata(
    tmp_path,
):
    input_path = (
        tmp_path / "input.jsonl"
    )

    output_path = (
        tmp_path / "output.jsonl"
    )

    record = {
        "id": "fineweb-00000001",
        "text": (
            "This is a valid AMOR "
            "training document."
        ),
        "source": "fineweb",
        "dataset_id": (
            "HuggingFaceFW/fineweb"
        ),
        "config": "sample-10BT",
        "split": "train",
        "token_count": 6,
    }

    input_path.write_text(
        json.dumps(record)
        + "\n",
        encoding="utf-8",
    )

    process_jsonl(
        str(input_path),
        str(output_path),
    )

    result = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert result["id"] == record["id"]

    assert result["source"] == (
        "fineweb"
    )

    assert result["dataset_id"] == (
        "HuggingFaceFW/fineweb"
    )

    assert result["config"] == (
        "sample-10BT"
    )

    assert result["split"] == "train"

    assert result["text_hash"]
    assert len(result["text_hash"]) == 64

    assert result["token_count"] > 0


def test_process_jsonl_repairs_encoding(
    tmp_path,
):
    input_path = (
        tmp_path / "input.jsonl"
    )

    output_path = (
        tmp_path / "output.jsonl"
    )

    record = {
        "id": "encoding-1",
        "text": (
            "AMOR Â· Training Â© 2026. "
            "This is a valid document."
        ),
        "token_count": 10,
    }

    input_path.write_text(
        json.dumps(record)
        + "\n",
        encoding="utf-8",
    )

    stats = process_jsonl(
        str(input_path),
        str(output_path),
    )

    assert stats["accepted_records"] == 1

    result = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert result["text"] == (
        "AMOR · Training © 2026. "
        "This is a valid document."
    )

    assert "Â" not in result["text"]


def test_process_jsonl_removes_duplicates(
    tmp_path,
):
    input_path = (
        tmp_path / "input.jsonl"
    )

    output_path = (
        tmp_path / "output.jsonl"
    )

    records = [
        {
            "id": "document-1",
            "text": (
                "This is an AMOR "
                "training document."
            ),
            "token_count": 7,
        },
        {
            "id": "document-2",
            "text": (
                "  This is an AMOR     "
                "training document.  "
            ),
            "token_count": 7,
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

    stats = process_jsonl(
        str(input_path),
        str(output_path),
    )

    assert stats["total_records"] == 2
    assert stats["accepted_records"] == 1
    assert stats["rejected_records"] == 0
    assert stats["duplicate_records"] == 1
    assert stats["unique_records"] == 1

    assert stats[
        "rejection_reasons"
    ] == {
        "duplicate": 1
    }

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        output = [
            json.loads(line)
            for line in file
        ]

    assert len(output) == 1


def test_process_jsonl_same_text_has_same_hash(
    tmp_path,
):
    input_path = (
        tmp_path / "input.jsonl"
    )

    output_path = (
        tmp_path / "output.jsonl"
    )

    record = {
        "id": "hash-test",
        "text": (
            "AMOR produces reliable "
            "language models."
        ),
        "token_count": 6,
    }

    input_path.write_text(
        json.dumps(record)
        + "\n",
        encoding="utf-8",
    )

    process_jsonl(
        str(input_path),
        str(output_path),
    )

    result = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    first_hash = result[
        "text_hash"
    ]

    assert isinstance(
        first_hash,
        str,
    )

    assert len(first_hash) == 64

    # SHA-256 hashes are hexadecimal.
    assert all(
        character in "0123456789abcdef"
        for character in first_hash
    )


def test_process_jsonl_reports_rejection_reason(
    tmp_path,
):
    input_path = (
        tmp_path / "input.jsonl"
    )

    output_path = (
        tmp_path / "output.jsonl"
    )

    record = {
        "id": "bad",
        "text": "",
        "token_count": 0,
    }

    input_path.write_text(
        json.dumps(record)
        + "\n",
        encoding="utf-8",
    )

    stats = process_jsonl(
        str(input_path),
        str(output_path),
    )

    assert stats[
        "rejection_reasons"
    ] == {
        "empty": 1
    }


def test_process_jsonl_rejects_non_string_text(
    tmp_path,
):
    input_path = (
        tmp_path / "input.jsonl"
    )

    output_path = (
        tmp_path / "output.jsonl"
    )

    record = {
        "id": "invalid-text",
        "text": 12345,
        "token_count": 1,
    }

    input_path.write_text(
        json.dumps(record)
        + "\n",
        encoding="utf-8",
    )

    stats = process_jsonl(
        str(input_path),
        str(output_path),
    )

    assert stats["total_records"] == 1
    assert stats["accepted_records"] == 0
    assert stats["rejected_records"] == 1
    assert stats["duplicate_records"] == 0

    assert stats[
        "rejection_reasons"
    ] == {
        "invalid_text": 1
    }


def test_process_jsonl_empty_input(
    tmp_path,
):
    input_path = (
        tmp_path / "input.jsonl"
    )

    output_path = (
        tmp_path / "output.jsonl"
    )

    input_path.write_text(
        "",
        encoding="utf-8",
    )

    stats = process_jsonl(
        str(input_path),
        str(output_path),
    )

    assert stats["total_records"] == 0
    assert stats["accepted_records"] == 0
    assert stats["rejected_records"] == 0
    assert stats["duplicate_records"] == 0
    assert stats["unique_records"] == 0
    assert stats["acceptance_rate"] == 0.0
    assert stats["input_tokens"] == 0
    assert stats["output_tokens"] == 0