import pytest

from src.amor.data.dedup import (
    deduplicate_records,
    text_hash,
)


def test_text_hash_is_deterministic():
    text = "Hello AMOR."

    assert text_hash(text) == text_hash(text)


def test_text_hash_changes_for_different_text():
    assert text_hash(
        "Hello AMOR."
    ) != text_hash(
        "Hello world."
    )


def test_text_hash_rejects_non_string():
    with pytest.raises(TypeError):
        text_hash(123)


def test_duplicate_records_are_removed():
    records = [
        {
            "id": "1",
            "text": "Hello AMOR.",
        },
        {
            "id": "2",
            "text": "Hello AMOR.",
        },
        {
            "id": "3",
            "text": "Different document.",
        },
    ]

    unique, stats = deduplicate_records(
        records
    )

    output = list(unique)

    assert len(output) == 2

    assert output[0]["id"] == "1"
    assert output[1]["id"] == "3"

    assert stats["total_records"] == 3
    assert stats["unique_records"] == 2
    assert stats["duplicate_records"] == 1


def test_first_occurrence_is_preserved():
    records = [
        {
            "id": "first",
            "text": "Same document.",
        },
        {
            "id": "second",
            "text": "Same document.",
        },
    ]

    unique, stats = deduplicate_records(
        records
    )

    output = list(unique)

    assert len(output) == 1
    assert output[0]["id"] == "first"


def test_unique_records_are_preserved():
    records = [
        {
            "id": "1",
            "text": "Document one.",
        },
        {
            "id": "2",
            "text": "Document two.",
        },
        {
            "id": "3",
            "text": "Document three.",
        },
    ]

    unique, stats = deduplicate_records(
        records
    )

    output = list(unique)

    assert len(output) == 3
    assert stats["total_records"] == 3
    assert stats["unique_records"] == 3
    assert stats["duplicate_records"] == 0


def test_text_hash_is_attached_to_output():
    records = [
        {
            "id": "1",
            "text": "Hello AMOR.",
        }
    ]

    unique, _ = deduplicate_records(
        records
    )

    output = list(unique)

    assert "text_hash" in output[0]

    assert output[0]["text_hash"] == (
        text_hash("Hello AMOR.")
    )