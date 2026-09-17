"""Unit tests for the dataset loader, including error paths."""

from pathlib import Path

import pytest

from llm_eval.storage.dataset_loader import DatasetLoadError, load_dataset

SAMPLE_DATASET_PATH = Path("data/eval/sample_dataset.json")


def test_load_sample_dataset() -> None:
    dataset = load_dataset(SAMPLE_DATASET_PATH)
    assert dataset.name == "sample_qa"
    assert dataset.version == "v1"
    assert len(dataset.examples) == 6
    assert {e.id for e in dataset.examples} == {
        "qa_001", "qa_002", "qa_003", "qa_004", "qa_005", "qa_006"
    }


def test_load_dataset_missing_file() -> None:
    with pytest.raises(DatasetLoadError, match="not found"):
        load_dataset("data/eval/does_not_exist.json")


def test_load_dataset_invalid_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="not valid JSON"):
        load_dataset(bad_file)


def test_load_dataset_schema_violation(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_schema.json"
    bad_file.write_text('{"name": "x", "version": "v1", "examples": []}', encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="schema validation"):
        load_dataset(bad_file)
