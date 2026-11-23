"""Dataset loader and validator for HR Policy RAG evaluation benchmarks."""

import json
import logging
from pathlib import Path
from typing import Union

from app.core.exceptions import AppException
from app.modules.evaluation.schemas import EvaluationDataset

logger = logging.getLogger(__name__)


def load_evaluation_dataset(file_path: Union[str, Path]) -> EvaluationDataset:
    """Load and validate an evaluation benchmark dataset from a JSON file.

    Args:
        file_path: Path to the JSON dataset file.

    Returns:
        Validated EvaluationDataset model instance.

    Raises:
        AppException: If file is missing, unreadable, or fails Pydantic schema validation.
    """
    path = Path(file_path)
    if not path.is_file():
        raise AppException(
            message=f"Evaluation dataset file not found: {path}",
            error_code="DATASET_NOT_FOUND",
            status_code=404,
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise AppException(
            message=f"Failed to read evaluation dataset JSON: {exc}",
            error_code="DATASET_READ_ERROR",
            status_code=400,
        ) from exc

    try:
        dataset = EvaluationDataset.model_validate(data)
        logger.info("Successfully loaded evaluation dataset v%s with %d test cases", dataset.version, len(dataset.test_cases))
        return dataset
    except Exception as exc:
        raise AppException(
            message=f"Evaluation dataset failed schema validation: {exc}",
            error_code="DATASET_VALIDATION_ERROR",
            status_code=422,
        ) from exc


def save_evaluation_dataset(dataset: EvaluationDataset, file_path: Union[str, Path]) -> None:
    """Save an EvaluationDataset to disk in JSON format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(mode="json"), f, indent=2, default=str)
    logger.info("Saved evaluation dataset to %s", path)
