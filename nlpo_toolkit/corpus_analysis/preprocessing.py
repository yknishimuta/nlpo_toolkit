from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.cleaner_contracts import (
    CleanerApplicationError,
    CleanerExecutionRequest,
)

from .corpus_errors import (
    CleanerExecutionError,
    CleanerInspectionError,
)
from .planning.models import CleanerPlan
from .ports import CorpusPreparationDependencies


def execute_preprocess(
    plan: CleanerPlan | None,
    *,
    dependencies: CorpusPreparationDependencies,
) -> Path | None:
    if plan is None:
        return None
    if not plan.config_path.exists():
        raise CleanerInspectionError(f"Cleaner config file not found: {plan.config_path}")
    try:
        result = dependencies.execute_cleaner(
            CleanerExecutionRequest(inspection=plan.inspection)
        )
    except CleanerApplicationError as exc:
        raise CleanerExecutionError(
            f"Cleaner preprocessing failed: {plan.config_path}: {exc}"
        ) from exc
    expected = plan.inspection.config.output_path.resolve()
    if result.configured_output_path != expected:
        raise CleanerExecutionError(
            f"Cleaner returned an unexpected output path: expected={expected}; "
            f"actual={result.configured_output_path}"
        )
    return result.configured_output_path
