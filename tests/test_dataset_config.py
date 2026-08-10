from src.amor.data.acquisition.config import (
    DATASET_SOURCES,
    DatasetSource,
)


def test_dataset_sources_exist():
    assert len(DATASET_SOURCES) == 5


def test_dataset_sources_are_valid():
    for source in DATASET_SOURCES:
        assert isinstance(
            source,
            DatasetSource,
        )

        assert source.name
        assert source.dataset_id
        assert source.split

        assert source.target_tokens > 0


def test_dataset_names_are_unique():
    names = [
        source.name
        for source in DATASET_SOURCES
    ]

    assert len(names) == len(set(names))


def test_dataset_ids_are_unique():
    dataset_ids = [
        source.dataset_id
        for source in DATASET_SOURCES
    ]

    assert len(dataset_ids) == len(set(dataset_ids))


def test_target_token_budget():
    total_tokens = sum(
        source.target_tokens
        for source in DATASET_SOURCES
    )

    assert total_tokens == 60_000_000