"""Loading and validating evaluation datasets from disk."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from llm_eval.evaluation.schemas import EvalDataset


class DatasetLoadError(Exception):
    """Raised when a dataset file is missing, malformed, or fails validation."""


def load_dataset(path: str | Path) -> EvalDataset:
    """Load and validate an EvalDataset from a JSON file.

    Raises:
        DatasetLoadError: if the file is missing, is not valid JSON, or does
            not match the EvalDataset schema. The original cause is chained
            so the underlying error is never silently lost.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise DatasetLoadError(f"Dataset file not found: {file_path}")

    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(f"Dataset file is not valid JSON: {file_path}") from exc

    try:
        return EvalDataset.model_validate(raw)
    except ValidationError as exc:
        message = f"Dataset file failed schema validation: {file_path}\n{exc}"
        raise DatasetLoadError(message) from exc
