from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from nlpo_toolkit.cleaner_contracts import CleanerConfigInspection
from nlpo_toolkit.comparison.configured import ComparisonSpec

from .config import AnalysisUnit, AppConfig, GroupingMode
from .config_references import ResolvedConfigFiles, resolve_config_files
from .corpus import (
    CleanerPlan,
    CorpusWorkItem,
    execute_preprocess,
    inspect_preprocess,
    resolve_cleaner_plan,
    resolve_corpus_work_items,
)
from .dependencies import CorpusPlanningDependencies
from .partition_models import PartitionSpec


@dataclass(frozen=True)
class AnalysisPlan:
    project_root: Path
    config_path: Path
    config: AppConfig
    cleaned_dir: Path | None
    grouping_mode: GroupingMode
    work_items: tuple[CorpusWorkItem, ...]
    group_files: Mapping[str, tuple[Path, ...]]
    cleaner_inspection: CleanerConfigInspection | None = None
    config_files: ResolvedConfigFiles = ResolvedConfigFiles()

    def __post_init__(self) -> None:
        immutable_group_files = MappingProxyType(
            {name: tuple(files) for name, files in self.group_files.items()}
        )
        object.__setattr__(self, "work_items", tuple(self.work_items))
        object.__setattr__(self, "group_files", immutable_group_files)

    @property
    def per_file(self) -> bool:
        return self.grouping_mode == "per_file"

    @property
    def auto_mode(self) -> bool:
        return self.grouping_mode == "auto_single_cleaned"

    @property
    def out_dir(self) -> Path:
        return resolve_out_dir(self.config, self.project_root)

    @property
    def auto_group_name(self) -> str:
        return self.config.grouping.auto_group_name

    @property
    def partition_specs(self) -> tuple[PartitionSpec, ...]:
        return self.config.validations.partitions

    @property
    def comparison_specs(self) -> tuple[ComparisonSpec, ...]:
        return self.config.comparisons

    @property
    def analysis_unit(self) -> AnalysisUnit:
        return resolve_analysis_unit(self.config)[0]

    @property
    def use_lemma(self) -> bool:
        return resolve_analysis_unit(self.config)[1]

    @property
    def csv_header(self) -> tuple[str, str]:
        return resolve_analysis_unit(self.config)[2]


class AnalysisPlanError(ValueError):
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


def resolve_out_dir(config: AppConfig, project_root: Path) -> Path:
    out_dir = Path(config.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    return out_dir


def ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_analysis_unit(
    config: AppConfig,
) -> tuple[AnalysisUnit, bool, tuple[str, str]]:
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
        raise AnalysisPlanError(
            "validations.partitions cannot be used with --group-by-file or grouping.mode: per_file"
        )
    if comparison_specs and per_file:
        raise AnalysisPlanError("comparisons cannot be used with grouping.mode=per_file")


def validate_partition_group_references(
    *,
    partition_specs: Sequence[PartitionSpec],
    group_files: Mapping[str, Sequence[Path]],
) -> None:
    for spec in partition_specs:
        for name in (spec.whole, *spec.parts):
            if not group_files.get(name):
                raise AnalysisPlanError(
                    f"Partition {spec.name} references empty group: {name}"
                )


def validate_comparison_group_references(
    *,
    comparison_specs: Sequence[ComparisonSpec],
    group_files: Mapping[str, Sequence[Path]],
) -> None:
    for spec in comparison_specs:
        for name in (spec.group_a, spec.group_b):
            if not group_files.get(name):
                raise AnalysisPlanError(
                    f"comparison {spec.name} references empty group: {name}"
                )


def _inspect_cleaner_plan(
    *,
    config: AppConfig,
    project_root: Path,
    dependencies: CorpusPlanningDependencies,
) -> tuple[CleanerPlan | None, CleanerConfigInspection | None]:
    plan = resolve_cleaner_plan(
        config,
        project_root,
        inspector=dependencies.cleaner_inspector,
    )
    return plan, plan.inspection if plan is not None else None


def build_analysis_plan(
    *,
    project_root: Path | None,
    script_dir: Path | None,
    config_path: Path,
    group_by_file: bool | None,
    auto_single_cleaned: bool,
    error_on_empty_group: bool,
    dependencies: CorpusPlanningDependencies,
    preprocess_mode: Literal["inspect", "execute"],
) -> AnalysisPlan:
    resolved_root, resolved_config = resolve_run_paths(
        project_root=project_root,
        script_dir=script_dir,
        config_path=config_path,
    )
    config = dependencies.load_config(resolved_config)
    if not config.groups:
        raise AnalysisPlanError("config.groups must be a non-empty mapping")

    cleaner_plan, cleaner_inspection = _inspect_cleaner_plan(
        config=config,
        project_root=resolved_root,
        dependencies=dependencies,
    )
    config_files = resolve_config_files(
        config=config,
        config_path=resolved_config,
        project_root=resolved_root,
        cleaner_inspection=cleaner_inspection,
    )
    if preprocess_mode == "inspect":
        cleaned_dir = inspect_preprocess(cleaner_plan)
    elif preprocess_mode == "execute":
        cleaned_dir = execute_preprocess(
            cleaner_plan,
            cleaner_loader=dependencies.cleaner_loader,
        )
    else:
        raise ValueError("preprocess_mode must be 'inspect' or 'execute'")

    resolved = resolve_corpus_work_items(
        config=config,
        project_root=resolved_root,
        cleaned_dir=cleaned_dir,
        group_by_file=bool(group_by_file),
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
    )
    return AnalysisPlan(
        project_root=resolved_root,
        config_path=resolved_config,
        config=config,
        cleaned_dir=cleaned_dir,
        grouping_mode=resolved.mode,
        work_items=resolved.work_items,
        group_files=resolved.group_files,
        cleaner_inspection=cleaner_inspection,
        config_files=config_files,
    )


def validate_count_plan(
    plan: AnalysisPlan,
    *,
    validate_references: bool = True,
) -> None:
    validate_specs_against_grouping(
        partition_specs=plan.partition_specs,
        comparison_specs=plan.comparison_specs,
        per_file=plan.per_file,
    )
    if validate_references:
        validate_partition_group_references(
            partition_specs=plan.partition_specs,
            group_files=plan.group_files,
        )
        validate_comparison_group_references(
            comparison_specs=plan.comparison_specs,
            group_files=plan.group_files,
        )


def build_count_plan(
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
) -> AnalysisPlan:
    plan = build_analysis_plan(
        project_root=project_root,
        script_dir=script_dir,
        config_path=config_path,
        group_by_file=group_by_file,
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
        dependencies=dependencies,
        preprocess_mode=preprocess_mode,
    )
    validate_count_plan(plan, validate_references=validate_references)
    return plan
