from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from nlpo_toolkit.cleaner_contracts import CleanerConfigInspection
from nlpo_toolkit.comparison.configured import ComparisonSpec

from .config import AnalysisUnit, AppConfig, GroupingMode
from .config_references import ResolvedConfigFiles, resolve_config_files
from .corpus import CorpusWorkItem, resolve_corpus_work_items
from .ports import CorpusPlanningDependencies, CorpusPreparationDependencies
from .preprocessing import CleanerPlan, prepare_analysis_plan, resolve_cleaner_plan
from .partition_models import PartitionSpec
from .requests import CorpusPreparationRequest, GroupingOverride


@dataclass(frozen=True)
class AnalysisPlan:
    project_root: Path
    config_path: Path
    config: AppConfig
    grouping_mode: GroupingMode
    error_on_empty_group: bool
    cleaner_plan: CleanerPlan | None
    cleaner_inspection: CleanerConfigInspection | None = None
    config_files: ResolvedConfigFiles = ResolvedConfigFiles()

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


@dataclass(frozen=True)
class ResolvedAnalysisPlan:
    definition: AnalysisPlan
    cleaned_dir: Path | None
    work_items: tuple[CorpusWorkItem, ...]
    group_files: Mapping[str, tuple[Path, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_items", tuple(self.work_items))
        object.__setattr__(
            self,
            "group_files",
            MappingProxyType(
                {name: tuple(files) for name, files in self.group_files.items()}
            ),
        )

    @property
    def project_root(self) -> Path:
        return self.definition.project_root

    @property
    def config_path(self) -> Path:
        return self.definition.config_path

    @property
    def config(self) -> AppConfig:
        return self.definition.config

    @property
    def config_files(self) -> ResolvedConfigFiles:
        return self.definition.config_files

    @property
    def cleaner_inspection(self) -> CleanerConfigInspection | None:
        return self.definition.cleaner_inspection

    @property
    def grouping_mode(self) -> GroupingMode:
        return self.definition.grouping_mode

    @property
    def per_file(self) -> bool:
        return self.definition.per_file

    @property
    def auto_mode(self) -> bool:
        return self.definition.auto_mode

    @property
    def auto_group_name(self) -> str:
        return self.definition.auto_group_name

    @property
    def out_dir(self) -> Path:
        return self.definition.out_dir

    @property
    def partition_specs(self) -> tuple[PartitionSpec, ...]:
        return self.definition.partition_specs

    @property
    def comparison_specs(self) -> tuple[ComparisonSpec, ...]:
        return self.definition.comparison_specs

    @property
    def analysis_unit(self) -> AnalysisUnit:
        return self.definition.analysis_unit

    @property
    def use_lemma(self) -> bool:
        return self.definition.use_lemma

    @property
    def csv_header(self) -> tuple[str, str]:
        return self.definition.csv_header


class AnalysisPlanError(ValueError):
    pass


def resolve_run_paths(
    *,
    project_root: Path,
    config_path: Path,
) -> tuple[Path, Path]:
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


def resolve_analysis_unit(
    config: AppConfig,
) -> tuple[AnalysisUnit, bool, tuple[str, str]]:
    unit = config.analysis_unit
    use_lemma = unit == "lemma"
    header = ("lemma", "count") if use_lemma else ("word", "frequency")
    if config.csv_header is not None:
        header = config.csv_header
    return unit, use_lemma, header


def resolve_effective_grouping_mode(
    *,
    configured_mode: GroupingMode,
    override: GroupingOverride | None,
) -> GroupingMode:
    if override == "auto_single_cleaned":
        return "auto_single_cleaned"
    if override == "per_file":
        return "per_file"
    return configured_mode


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
    request: CorpusPreparationRequest,
    *,
    dependencies: CorpusPlanningDependencies,
) -> AnalysisPlan:
    resolved_root, resolved_config = resolve_run_paths(
        project_root=request.project_root,
        config_path=request.config_path,
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
    return AnalysisPlan(
        project_root=resolved_root,
        config_path=resolved_config,
        config=config,
        grouping_mode=resolve_effective_grouping_mode(
            configured_mode=config.grouping.mode,
            override=request.grouping_override,
        ),
        error_on_empty_group=request.error_on_empty_group,
        cleaner_plan=cleaner_plan,
        cleaner_inspection=cleaner_inspection,
        config_files=config_files,
    )


def resolve_analysis_inputs(
    plan: AnalysisPlan,
    *,
    cleaned_dir: Path | None,
) -> ResolvedAnalysisPlan:
    resolved = resolve_corpus_work_items(
        config=plan.config,
        project_root=plan.project_root,
        cleaned_dir=cleaned_dir,
        grouping_mode=plan.grouping_mode,
        error_on_empty_group=plan.error_on_empty_group,
    )
    return ResolvedAnalysisPlan(
        definition=plan,
        cleaned_dir=cleaned_dir,
        work_items=resolved.work_items,
        group_files=resolved.group_files,
    )


def validate_count_plan_structure(plan: AnalysisPlan) -> None:
    validate_specs_against_grouping(
        partition_specs=plan.partition_specs,
        comparison_specs=plan.comparison_specs,
        per_file=plan.per_file,
    )


def validate_count_group_references(plan: ResolvedAnalysisPlan) -> None:
    validate_partition_group_references(
        partition_specs=plan.partition_specs,
        group_files=plan.group_files,
    )
    validate_comparison_group_references(
        comparison_specs=plan.comparison_specs,
        group_files=plan.group_files,
    )


def build_count_plan(
    request: CorpusPreparationRequest,
    *,
    dependencies: CorpusPlanningDependencies,
) -> AnalysisPlan:
    plan = build_analysis_plan(
        request,
        dependencies=dependencies,
    )
    validate_count_plan_structure(plan)
    return plan


def prepare_count_plan(
    plan: AnalysisPlan,
    *,
    dependencies: CorpusPreparationDependencies,
) -> ResolvedAnalysisPlan:
    resolved = prepare_analysis_plan(plan, dependencies=dependencies)
    validate_count_group_references(resolved)
    return resolved
