from collections.abc import Iterable, Iterator
import hashlib


def text_hash(text: str) -> str:
    """
    Return a deterministic SHA-256 hash for text.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def deduplicate_records(
    records: Iterable[dict],
) -> tuple[Iterator[dict], dict]:
    """
    Remove exact duplicate documents.

    The first occurrence of each unique document
    is retained.

    The returned statistics dictionary is updated
    as the iterator is consumed.
    """

    seen: set[str] = set()

    stats = {
        "total_records": 0,
        "unique_records": 0,
        "duplicate_records": 0,
    }

    def generate() -> Iterator[dict]:
        for record in records:
            stats["total_records"] += 1

            text = record.get("text", "")

            if not isinstance(text, str):
                stats["duplicate_records"] += 1
                continue

            digest = text_hash(text)

            if digest in seen:
                stats["duplicate_records"] += 1
                continue

            seen.add(digest)

            stats["unique_records"] += 1

            output = dict(record)
            output["text_hash"] = digest

            yield output

    return generate(), stats