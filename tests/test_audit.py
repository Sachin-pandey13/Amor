import json

from src.amor.data.audit import audit_jsonl


def write_jsonl(path, records):
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_audit_counts_documents_and_tokens(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": "Hello AMOR.",
            "token_count": 3,
            "source": "fineweb",
        },
        {
            "id": "2",
            "text": "Python is useful.",
            "token_count": 4,
            "source": "stackv2",
        },
    ]

    write_jsonl(
        input_path,
        records,
    )

    stats = audit_jsonl(
        str(input_path)
    )

    assert stats["total_records"] == 2
    assert stats["total_tokens"] == 7


def test_audit_source_distribution(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": "A",
            "token_count": 1,
            "source": "fineweb",
        },
        {
            "id": "2",
            "text": "B",
            "token_count": 1,
            "source": "fineweb",
        },
        {
            "id": "3",
            "text": "C",
            "token_count": 1,
            "source": "stackv2",
        },
    ]

    write_jsonl(
        input_path,
        records,
    )

    stats = audit_jsonl(
        str(input_path)
    )

    assert stats["source_distribution"] == {
        "fineweb": 2,
        "stackv2": 1,
    }


def test_audit_detects_empty_documents(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": "",
            "token_count": 0,
        },
        {
            "id": "2",
            "text": "Valid document.",
            "token_count": 3,
        },
    ]

    write_jsonl(
        input_path,
        records,
    )

    stats = audit_jsonl(
        str(input_path)
    )

    assert stats["empty_documents"] == 1


def test_audit_detects_short_documents(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": "Hi",
            "token_count": 1,
        },
        {
            "id": "2",
            "text": (
                "This is a sufficiently "
                "long document."
            ),
            "token_count": 6,
        },
    ]

    write_jsonl(
        input_path,
        records,
    )

    stats = audit_jsonl(
        str(input_path),
        min_tokens=5,
    )

    assert stats["short_documents"] == 1


def test_audit_document_length_statistics(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": "A",
            "token_count": 2,
        },
        {
            "id": "2",
            "text": "B",
            "token_count": 5,
        },
        {
            "id": "3",
            "text": "C",
            "token_count": 8,
        },
    ]

    write_jsonl(
        input_path,
        records,
    )

    stats = audit_jsonl(
        str(input_path)
    )

    assert stats["min_tokens"] == 2
    assert stats["max_tokens"] == 8
    assert stats["average_tokens"] == 5.0


def test_audit_detects_duplicate_hashes(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": "Same text.",
            "token_count": 3,
            "text_hash": "abc",
        },
        {
            "id": "2",
            "text": "Same text.",
            "token_count": 3,
            "text_hash": "abc",
        },
        {
            "id": "3",
            "text": "Different text.",
            "token_count": 4,
            "text_hash": "xyz",
        },
    ]

    write_jsonl(
        input_path,
        records,
    )

    stats = audit_jsonl(
        str(input_path)
    )

    assert stats["duplicate_hashes"] == 1


def test_audit_character_statistics(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": "Hello world!",
            "token_count": 3,
        },
        {
            "id": "2",
            "text": "Python code.",
            "token_count": 3,
        },
    ]

    write_jsonl(
        input_path,
        records,
    )

    stats = audit_jsonl(
        str(input_path)
    )

    assert stats["total_characters"] == (
        len("Hello world!")
        + len("Python code.")
    )


def test_audit_counts_missing_metadata(
    tmp_path,
):
    input_path = (
        tmp_path / "corpus.jsonl"
    )

    records = [
        {
            "id": "1",
            "text": "Document one.",
            "token_count": 3,
        },
        {
            "id": "2",
            "text": "Document two.",
            "token_count": 3,
            "source": "fineweb",
        },
    ]

    write_jsonl(
        input_path,
        records,
    )

    stats = audit_jsonl(
        str(input_path)
    )

    assert stats["missing_source"] == 1


def test_audit_empty_corpus(
    tmp_path,
):
    input_path = (
        tmp_path / "empty.jsonl"
    )

    input_path.write_text(
        "",
        encoding="utf-8",
    )

    stats = audit_jsonl(
        str(input_path)
    )

    assert stats["total_records"] == 0
    assert stats["total_tokens"] == 0
    assert stats["average_tokens"] == 0.0
    assert stats["min_tokens"] == 0
    assert stats["max_tokens"] == 0


def test_audit_missing_file(
    tmp_path,
):
    input_path = (
        tmp_path / "missing.jsonl"
    )

    try:
        audit_jsonl(
            str(input_path)
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Expected FileNotFoundError"
        )