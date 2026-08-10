from collections.abc import Iterator
import json
from pathlib import Path

from datasets import load_dataset

from .config import DatasetSource


def stream_dataset(
    source: DatasetSource,
    max_documents: int,
) -> Iterator[dict]:
    """
    Stream documents from a Hugging Face dataset.

    No full dataset is downloaded into memory.
    """

    if max_documents <= 0:
        raise ValueError(
            "max_documents must be greater than zero."
        )

    dataset = load_dataset(
        source.dataset_id,
        name=source.config,
        split=source.split,
        streaming=True,
    )

    for index, record in enumerate(dataset):
        if index >= max_documents:
            break

        text = extract_text(record)

        if not text:
            continue

        yield {
            "id": f"{source.name}-{index:08d}",
            "text": text,
            "source": source.name,
            "dataset_id": source.dataset_id,
            "config": source.config,
            "split": source.split,
        }


def extract_instruction_pair(
    record: dict,
) -> str:
    """
    Convert an instruction/response record into
    a single training example.

    Used by datasets such as Aya.
    """

    inputs = record.get("inputs")
    targets = record.get("targets")

    if not isinstance(inputs, str):
        return ""

    if not isinstance(targets, str):
        return ""

    inputs = inputs.strip()
    targets = targets.strip()

    if not inputs or not targets:
        return ""

    return (
        f"User:\n{inputs}\n\n"
        f"Assistant:\n{targets}"
    )


def extract_text(record: dict) -> str:
    """
    Extract textual content from a dataset record.

    Instruction/response datasets are handled first.
    """

    instruction_text = extract_instruction_pair(
        record
    )

    if instruction_text:
        return instruction_text

    possible_fields = (
        "text",
        "content",
        "document",
        "completion",
        "response",
    )

    for field in possible_fields:
        value = record.get(field)

        if isinstance(value, str):
            text = value.strip()

            if text:
                return text

    return ""


def save_jsonl(
    records: Iterator[dict],
    output_path: str,
) -> int:
    """
    Save streamed records to JSONL.
    """

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with output.open(
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

            count += 1

    return count