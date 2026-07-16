from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nlpo_toolkit.cleaner_contracts import CleanerConfigError

from ..config.models import AppConfig, GroupingMode
from ..config_references import resolve_config_files
from ..corpus import resolve_project_path
from ..ports import CorpusPlanningDependencies
from ..requests import CorpusPreparationRequest, GroupingOverride
from .models import AnalysisMode, AnalysisPlan, CleanerPlan
from .validate import (
    AnalysisPlanError,
    validate_analysis_config,
    validate_count_plan_structure,
)


@dataclass(frozen=True)
class _ResolvedRequestPaths:
    project_root: Path
    config_path: Path


def _resolve_request_paths(request: CorpusPreparationRequest) -> _ResolvedRequestPaths:
    project_root = Path(request.project_root).resolve()
    config_path = Path(request.config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return _ResolvedRequestPaths(project_root, config_path)


def _resolve_out_dir(config: AppConfig, project_root: Path) -> Path:
    path = Path(config.out_dir)
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def _build_analysis_mode(config: AppConfig) -> AnalysisMode:
    default_header = (
        ("lemma", "count")
        if config.analysis_unit == "lemma"
        else ("word", "frequency")
    )
    return AnalysisMode(
        unit=config.analysis_unit,
        csv_header=config.csv_header or default_header,
    )


def _effective_grouping_mode(
    configured: GroupingMode, override: GroupingOverride | None
) -> GroupingMode:
    if override in {"auto_single_cleaned", "per_file"}:
        return override
    return configured


def _build_cleaner_plan(
    *,
    config: AppConfig,
    project_root: Path,
    dependencies: CorpusPlanningDependencies,
) -> CleanerPlan | None:
    if config.preprocess.kind != "cleaner":
        return None
    if not config.preprocess.config:
        raise AnalysisPlanError(
            "'preprocess.config' is required when preprocess.kind=cleaner"
        )
    config_path = resolve_project_path(project_root, config.preprocess.config)
    try:
        inspection = dependencies.cleaner_inspector(config_path)
    except CleanerConfigError as exc:
        raise AnalysisPlanError(str(exc)) from exc
    return CleanerPlan(config_path, inspection)


def build_analysis_plan(
    request: CorpusPreparationRequest,
    *,
    dependencies: CorpusPlanningDependencies,
) -> AnalysisPlan:
    paths = _resolve_request_paths(request)
    config = dependencies.load_config(paths.config_path)
    validate_analysis_config(config)
    grouping_mode = _effective_grouping_mode(
        config.grouping.mode, request.grouping_override
    )
    analysis_mode = _build_analysis_mode(config)
    cleaner_plan = _build_cleaner_plan(
        config=config,
        project_root=paths.project_root,
        dependencies=dependencies,
    )
    config_files = resolve_config_files(
        config=config,
        config_path=paths.config_path,
        project_root=paths.project_root,
        cleaner_inspection=(
            cleaner_plan.inspection if cleaner_plan is not None else None
        ),
    )
    return AnalysisPlan(
        project_root=paths.project_root,
        config_path=paths.config_path,
        config=config,
        out_dir=_resolve_out_dir(config, paths.project_root),
        grouping_mode=grouping_mode,
        error_on_empty_group=request.error_on_empty_group,
        analysis_mode=analysis_mode,
        cleaner_plan=cleaner_plan,
        config_files=config_files,
    )


def build_count_plan(
    request: CorpusPreparationRequest,
    *,
    dependencies: CorpusPlanningDependencies,
) -> AnalysisPlan:
    plan = build_analysis_plan(request, dependencies=dependencies)
    validate_count_plan_structure(plan)
    return plan
