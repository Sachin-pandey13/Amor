import json

import pytest

from src.amor.data.acquisition.manifest import (
    DatasetManifest,
    load_manifest,
    save_manifest,
)


def create_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_name="fineweb",
        dataset_id="HuggingFaceFW/fineweb",
        config="sample-10BT",
        split="train",
        target_tokens=1000,
    )


def test_manifest_creation():
    manifest = create_manifest()

    assert manifest.dataset_name == "fineweb"
    assert manifest.dataset_id == (
        "HuggingFaceFW/fineweb"
    )
    assert manifest.target_tokens == 1000
    assert manifest.documents == 0
    assert manifest.actual_tokens == 0
    assert manifest.acquired_at


def test_manifest_defaults():
    manifest = create_manifest()

    assert manifest.documents == 0
    assert manifest.actual_tokens == 0
    assert manifest.source_license is None
    assert manifest.dataset_revision is None


def test_manifest_rejects_empty_name():
    with pytest.raises(ValueError):
        DatasetManifest(
            dataset_name="",
            dataset_id="example",
            config=None,
            split="train",
            target_tokens=1000,
        )


def test_manifest_rejects_empty_dataset_id():
    with pytest.raises(ValueError):
        DatasetManifest(
            dataset_name="example",
            dataset_id="",
            config=None,
            split="train",
            target_tokens=1000,
        )


def test_manifest_rejects_invalid_target():
    with pytest.raises(ValueError):
        DatasetManifest(
            dataset_name="example",
            dataset_id="example",
            config=None,
            split="train",
            target_tokens=0,
        )


def test_manifest_rejects_negative_documents():
    with pytest.raises(ValueError):
        DatasetManifest(
            dataset_name="example",
            dataset_id="example",
            config=None,
            split="train",
            target_tokens=1000,
            documents=-1,
        )


def test_manifest_rejects_negative_tokens():
    with pytest.raises(ValueError):
        DatasetManifest(
            dataset_name="example",
            dataset_id="example",
            config=None,
            split="train",
            target_tokens=1000,
            actual_tokens=-1,
        )


def test_save_manifest(tmp_path):
    manifest = create_manifest()

    manifest.documents = 10
    manifest.actual_tokens = 950

    output = (
        tmp_path / "manifest.json"
    )

    save_manifest(
        manifest,
        str(output),
    )

    assert output.exists()

    with output.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert data["dataset_name"] == "fineweb"
    assert data["documents"] == 10
    assert data["actual_tokens"] == 950


def test_load_manifest(tmp_path):
    manifest = create_manifest()

    manifest.documents = 20
    manifest.actual_tokens = 1000

    output = (
        tmp_path / "manifest.json"
    )

    save_manifest(
        manifest,
        str(output),
    )

    loaded = load_manifest(
        str(output)
    )

    assert loaded.dataset_name == (
        manifest.dataset_name
    )

    assert loaded.dataset_id == (
        manifest.dataset_id
    )

    assert loaded.documents == (
        manifest.documents
    )

    assert loaded.actual_tokens == (
        manifest.actual_tokens
    )


def test_manifest_round_trip(tmp_path):
    manifest = create_manifest()

    manifest.documents = 42
    manifest.actual_tokens = 987

    output = (
        tmp_path / "manifest.json"
    )

    save_manifest(
        manifest,
        str(output),
    )

    loaded = load_manifest(
        str(output)
    )

    assert loaded == manifest