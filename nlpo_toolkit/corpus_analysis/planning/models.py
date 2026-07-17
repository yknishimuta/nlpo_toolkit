from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from nlpo_toolkit.immutable_collections import freeze_tuple_mapping

from nlpo_toolkit.cleaner_contracts import CleanerConfigInspection

from ..config.models import AnalysisUnit, AppConfig, GroupingMode
from ..config_references import ResolvedConfigFiles
from ..corpus import CorpusWorkItem


@dataclass(frozen=True)
class AnalysisMode:
    unit: AnalysisUnit
    csv_header: tuple[str, str]

    @property
    def use_lemma(self) -> bool:
        return self.unit == "lemma"


@dataclass(frozen=True)
class CleanerPlan:
    config_path: Path
    inspection: CleanerConfigInspection

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", self.config_path.resolve())

    @property
    def output_path(self) -> Path:
        return self.inspection.config.output_path


@dataclass(frozen=True)
class AnalysisPlan:
    project_root: Path
    config_path: Path
    config: AppConfig
    out_dir: Path
    grouping_mode: GroupingMode
    error_on_empty_group: bool
    analysis_mode: AnalysisMode
    cleaner_plan: CleanerPlan | None
    config_files: ResolvedConfigFiles

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", self.project_root.resolve())
        object.__setattr__(self, "config_path", self.config_path.resolve())
        object.__setattr__(self, "out_dir", self.out_dir.resolve())

    @property
    def per_file(self) -> bool:
        return self.grouping_mode == "per_file"

    @property
    def auto_mode(self) -> bool:
        return self.grouping_mode == "auto_single_cleaned"


@dataclass(frozen=True)
class ResolvedAnalysisPlan:
    definition: AnalysisPlan
    cleaned_dir: Path | None
    work_items: tuple[CorpusWorkItem, ...]
    group_files: Mapping[str, tuple[Path, ...]]

    def __post_init__(self) -> None:
        if self.cleaned_dir is not None:
            object.__setattr__(self, "cleaned_dir", self.cleaned_dir.resolve())
        object.__setattr__(self, "work_items", tuple(self.work_items))
        object.__setattr__(self, "group_files", freeze_tuple_mapping(self.group_files))
