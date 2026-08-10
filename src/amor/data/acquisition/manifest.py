from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class DatasetManifest:
    """
    Metadata describing one acquired dataset source.
    """

    dataset_name: str
    dataset_id: str
    config: str | None
    split: str
    target_tokens: int

    documents: int = 0
    actual_tokens: int = 0

    source_license: str | None = None
    dataset_revision: str | None = None

    acquired_at: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError(
                "dataset_name cannot be empty."
            )

        if not self.dataset_id:
            raise ValueError(
                "dataset_id cannot be empty."
            )

        if not self.split:
            raise ValueError(
                "split cannot be empty."
            )

        if self.target_tokens <= 0:
            raise ValueError(
                "target_tokens must be greater than zero."
            )

        if self.documents < 0:
            raise ValueError(
                "documents cannot be negative."
            )

        if self.actual_tokens < 0:
            raise ValueError(
                "actual_tokens cannot be negative."
            )

        if not self.acquired_at:
            self.acquired_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )


def save_manifest(
    manifest: DatasetManifest,
    output_path: str,
) -> None:
    """
    Save a dataset manifest as JSON.
    """

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(manifest),
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_manifest(
    input_path: str,
) -> DatasetManifest:
    """
    Load a dataset manifest from JSON.
    """

    input_file = Path(input_path)

    with input_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return DatasetManifest(**data)