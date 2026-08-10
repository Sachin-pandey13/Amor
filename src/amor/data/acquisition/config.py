from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSource:
    """
    Configuration describing one external dataset source.
    """

    name: str
    dataset_id: str
    config: str | None = None
    split: str = "train"
    target_tokens: int = 0


# Initial AMOR-006 corpus sources.
DATASET_SOURCES = (
    DatasetSource(
        name="fineweb",
        dataset_id="HuggingFaceFW/fineweb",
        config="sample-10BT",
        target_tokens=30_000_000,
    ),
    DatasetSource(
        name="fineweb_edu",
        dataset_id="HuggingFaceFW/fineweb-edu",
        config="default",
        target_tokens=10_000_000,
    ),
    DatasetSource(
        name="stackv2",
        dataset_id="common-pile/stackv2_edu_filtered",
        config="default",
        target_tokens=10_000_000,
    ),
    DatasetSource(
        name="finemath",
        dataset_id="HuggingFaceTB/finemath",
        config="finemath-4plus",
        target_tokens=5_000_000,
    ),
    DatasetSource(
        name="aya",
        dataset_id="CohereLabs/aya_dataset",
        config="default",
        target_tokens=5_000_000,
    ),
)