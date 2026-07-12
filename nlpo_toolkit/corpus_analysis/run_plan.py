from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from nlpo_toolkit.comparison.configured import ComparisonSpec
from nlpo_toolkit.cleaner_contracts import CleanerConfigInspection
from .config import AppConfig
from .corpus import (
    CorpusWorkItem,
    execute_preprocess,
    inspect_preprocess,
    resolve_cleaner_plan,
    resolve_corpus_work_items,
)
from .dependencies import CorpusPlanningDependencies
from .partition_models import PartitionSpec


@dataclass(frozen=True)
class CorpusPlan:
    project_root: Path
    config_path: Path
    config: AppConfig
    cleaned_dir: Path | None
    grouping_mode: str
    per_file: bool
    auto_mode: bool
    work_items: tuple[CorpusWorkItem, ...]
    group_files: Mapping[str, tuple[Path, ...]]
    cleaner_inspection: CleanerConfigInspection | None = None


@dataclass(frozen=True)
class RunPlan:
    project_root: Path
    config_path: Path
    config: AppConfig
    out_dir: Path
    cleaned_dir: Path | None
    grouping_mode: str
    per_file: bool
    auto_mode: bool
    auto_group_name: str
    work_items: tuple[CorpusWorkItem, ...]
    group_files: Mapping[str, tuple[Path, ...]]
    partition_specs: tuple[PartitionSpec, ...]
    comparison_specs: tuple[ComparisonSpec, ...]
    analysis_unit: str
    use_lemma: bool
    csv_header: tuple[str, str]
    cleaner_inspection: CleanerConfigInspection | None = None


class RunPlanError(ValueError):
    pass


def resolve_run_paths(
    *,
    project_root: Path | None,
    script_dir: Path | None,
    config_path: Path,
) -> tuple[Path, Path]:
    if project_root is None:
        if script_dir is None:
            raise TypeError("project_root is required")
        project_root = script_dir

    resolved_root = Path(project_root).resolve()
    resolved_config = Path(config_path)
    if not resolved_config.is_absolute():
        resolved_config = (resolved_root / resolved_config).resolve()
    if not resolved_config.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config}")
    return resolved_root, resolved_config


def resolve_grouping_flags(
    *,
    config: AppConfig,
    group_by_file: bool | None,
    auto_single_cleaned: bool,
) -> tuple[str, bool, bool]:
    grouping_mode = config.grouping.mode
    auto_mode = bool(auto_single_cleaned) or grouping_mode == "auto_single_cleaned"
    per_file = (bool(group_by_file) or grouping_mode == "per_file") and not auto_mode
    return grouping_mode, per_file, auto_mode


def resolve_out_dir(config: AppConfig, project_root: Path) -> Path:
    out_dir = Path(config.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    return out_dir


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_analysis_unit(config: AppConfig) -> tuple[str, bool, tuple[str, str]]:
    unit = config.analysis_unit
    use_lemma = unit == "lemma"
    header = ("lemma", "count") if use_lemma else ("word", "frequency")
    if config.csv_header is not None:
        header = config.csv_header
    return unit, use_lemma, header


def validate_specs_against_grouping(
    *,
    partition_specs: Sequence[PartitionSpec],
    comparison_specs: Sequence[ComparisonSpec],
    per_file: bool,
) -> None:
    if partition_specs and per_file:
        raise RunPlanError(
            "validations.partitions cannot be used with --group-by-file or grouping.mode: per_file"
        )
    if comparison_specs and per_file:
        raise RunPlanError("comparisons cannot be used with grouping.mode=per_file")


def validate_partition_group_references(
    *,
    partition_specs: Sequence[PartitionSpec],
    group_files: Mapping[str, Sequence[Path]],
) -> None:
    for spec in partition_specs:
        for name in (spec.whole, *spec.parts):
            if not group_files.get(name):
                raise RunPlanError(f"Partition {spec.name} references empty group: {name}")


def validate_comparison_group_references(
    *,
    comparison_specs: Sequence[ComparisonSpec],
    group_files: Mapping[str, Sequence[Path]],
) -> None:
    for spec in comparison_specs:
        for name in (spec.group_a, spec.group_b):
            if not group_files.get(name):
                raise RunPlanError(f"comparison {spec.name} references empty group: {name}")


def _resolve_cleaned_dir(
    *,
    config: AppConfig,
    project_root: Path,
    preprocess_mode: Literal["inspect", "execute"],
    dependencies: CorpusPlanningDependencies,
) -> tuple[Path | None, CleanerConfigInspection | None]:
    plan = resolve_cleaner_plan(
        config,
        project_root,
        inspector=dependencies.cleaner_inspector,
    )
    if preprocess_mode == "inspect":
        return inspect_preprocess(plan), plan.inspection if plan is not None else None
    if preprocess_mode == "execute":
        if plan is None:
            return None, None
        return (
            execute_preprocess(plan, cleaner_loader=dependencies.cleaner_loader),
            plan.inspection,
        )
    raise ValueError("preprocess_mode must be 'inspect' or 'execute'")


def build_run_plan(
    *,
    project_root: Path | None,
    script_dir: Path | None,
    config_path: Path,
    group_by_file: bool | None,
    auto_single_cleaned: bool,
    error_on_empty_group: bool,
    dependencies: CorpusPlanningDependencies,
    preprocess_mode: Literal["inspect", "execute"],
    validate_references: bool = True,
) -> RunPlan:
    corpus_plan = build_corpus_plan(
        project_root=project_root,
        script_dir=script_dir,
        config_path=config_path,
        group_by_file=group_by_file,
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
        dependencies=dependencies,
        preprocess_mode=preprocess_mode,
    )
    config = corpus_plan.config
    partition_specs = config.partition_validations
    comparison_specs = config.comparisons
    validate_specs_against_grouping(
        partition_specs=partition_specs,
        comparison_specs=comparison_specs,
        per_file=corpus_plan.per_file,
    )
    if validate_references:
        validate_partition_group_references(
            partition_specs=partition_specs,
            group_files=corpus_plan.group_files,
        )
        validate_comparison_group_references(
            comparison_specs=comparison_specs,
            group_files=corpus_plan.group_files,
        )
    analysis_unit, use_lemma, csv_header = resolve_analysis_unit(config)

    return RunPlan(
        project_root=corpus_plan.project_root,
        config_path=corpus_plan.config_path,
        config=config,
        out_dir=resolve_out_dir(config, corpus_plan.project_root),
        cleaned_dir=corpus_plan.cleaned_dir,
        grouping_mode=corpus_plan.grouping_mode,
        per_file=corpus_plan.per_file,
        auto_mode=corpus_plan.auto_mode,
        auto_group_name=config.grouping.auto_group_name,
        work_items=corpus_plan.work_items,
        group_files=corpus_plan.group_files,
        partition_specs=partition_specs,
        comparison_specs=comparison_specs,
        analysis_unit=analysis_unit,
        use_lemma=use_lemma,
        csv_header=csv_header,
        cleaner_inspection=corpus_plan.cleaner_inspection,
    )


def build_corpus_plan(
    *,
    project_root: Path | None,
    script_dir: Path | None,
    config_path: Path,
    group_by_file: bool | None,
    auto_single_cleaned: bool,
    error_on_empty_group: bool,
    dependencies: CorpusPlanningDependencies,
    preprocess_mode: Literal["inspect", "execute"],
) -> CorpusPlan:
    resolved_root, resolved_config = resolve_run_paths(
        project_root=project_root,
        script_dir=script_dir,
        config_path=config_path,
    )
    config = dependencies.load_config(resolved_config)
    if not config.groups:
        raise RunPlanError("config.groups must be a non-empty mapping")

    grouping_mode, per_file, auto_mode = resolve_grouping_flags(
        config=config,
        group_by_file=group_by_file,
        auto_single_cleaned=auto_single_cleaned,
    )
    cleaned_dir, cleaner_inspection = _resolve_cleaned_dir(
        config=config,
        project_root=resolved_root,
        preprocess_mode=preprocess_mode,
        dependencies=dependencies,
    )
    resolved = resolve_corpus_work_items(
        config=config,
        project_root=resolved_root,
        cleaned_dir=cleaned_dir,
        group_by_file=per_file,
        auto_single_cleaned=auto_mode,
        error_on_empty_group=error_on_empty_group,
    )
    return CorpusPlan(
        project_root=resolved_root,
        config_path=resolved_config,
        config=config,
        cleaned_dir=cleaned_dir,
        grouping_mode=grouping_mode,
        per_file=per_file,
        auto_mode=auto_mode,
        work_items=tuple(resolved.work_items),
        group_files=resolved.group_files,
        cleaner_inspection=cleaner_inspection,
    )
