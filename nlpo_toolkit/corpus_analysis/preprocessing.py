from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nlpo_toolkit.cleaner_contracts import CleanerConfigError, CleanerConfigInspection

from .cleaner_runtime import run_cleaner
from .config import AppConfig
from .corpus import resolve_project_path
from .corpus_errors import CleanerInspectionError, CorpusPreparationError

if TYPE_CHECKING:
    from .ports import CleanerConfigInspector, CorpusPreparationDependencies
    from .run_plan import AnalysisPlan, ResolvedAnalysisPlan


@dataclass(frozen=True)
class CleanerPlan:
    config_path: Path
    inspection: CleanerConfigInspection


def resolve_cleaner_plan(
    config: AppConfig,
    project_root: Path,
    *,
    inspector: CleanerConfigInspector,
) -> CleanerPlan | None:
    if config.preprocess.kind != "cleaner":
        return None
    if not config.preprocess.config:
        raise CorpusPreparationError(
            "'preprocess.config' is required when preprocess.kind=cleaner"
        )
    config_path = resolve_project_path(project_root, config.preprocess.config)
    try:
        inspection = inspector(config_path)
    except CleanerConfigError as exc:
        raise CleanerInspectionError(str(exc)) from exc
    return CleanerPlan(config_path=config_path, inspection=inspection)


def execute_preprocess(
    plan: CleanerPlan | None,
    *,
    dependencies: CorpusPreparationDependencies,
) -> Path | None:
    if plan is None:
        return None
    if not plan.config_path.exists():
        raise CleanerInspectionError(f"Cleaner config file not found: {plan.config_path}")
    run_cleaner(
        config_path=plan.config_path,
        cleaner_loader=dependencies.cleaner_loader,
    )
    return plan.inspection.config.output_path


def inspect_analysis_plan(plan: AnalysisPlan) -> ResolvedAnalysisPlan:
    from .run_plan import resolve_analysis_inputs

    cleaned_dir = (
        plan.cleaner_plan.inspection.config.output_path
        if plan.cleaner_plan is not None
        else None
    )
    return resolve_analysis_inputs(plan, cleaned_dir=cleaned_dir)


def prepare_analysis_plan(
    plan: AnalysisPlan,
    *,
    dependencies: CorpusPreparationDependencies,
) -> ResolvedAnalysisPlan:
    from .run_plan import resolve_analysis_inputs

    cleaned_dir = execute_preprocess(plan.cleaner_plan, dependencies=dependencies)
    return resolve_analysis_inputs(plan, cleaned_dir=cleaned_dir)
