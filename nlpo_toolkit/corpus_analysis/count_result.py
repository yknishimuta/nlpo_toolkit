from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from nlpo_toolkit.immutable_collections import freeze_tuple_mapping

from .artifacts.models import ArtifactKind, ArtifactPlan
from .config_references import ConfigFileReference
from .partition_run_results import PartitionMismatchSummary
from .planning.models import ResolvedAnalysisPlan


@dataclass(frozen=True)
class CountRunResult:
    exit_code: int
    plan: ResolvedAnalysisPlan
    groups_files: Mapping[str, tuple[Path, ...]]
    input_files: tuple[Path, ...]
    cleaned_files: tuple[Path, ...]
    artifact_plan: ArtifactPlan
    config_references: tuple[ConfigFileReference, ...]
    partition_mismatches: tuple[PartitionMismatchSummary, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups_files", freeze_tuple_mapping(self.groups_files))
        object.__setattr__(self, "input_files", tuple(self.input_files))
        object.__setattr__(self, "cleaned_files", tuple(self.cleaned_files))
        object.__setattr__(self, "config_references", tuple(self.config_references))
        object.__setattr__(self, "partition_mismatches", tuple(self.partition_mismatches))

    @property
    def generated_outputs(self) -> tuple[Path, ...]:
        return self.artifact_plan.paths

    @property
    def trace_files(self) -> tuple[Path, ...]:
        return tuple(
            artifact.path
            for artifact in self.artifact_plan.select(
                kinds={ArtifactKind.DIAGNOSTIC_TRACE}
            )
        )

    @property
    def output_files(self) -> tuple[Path, ...]:
        return tuple(
            artifact.path
            for artifact in self.artifact_plan.artifacts
            if artifact.kind is not ArtifactKind.DIAGNOSTIC_TRACE
        )

    @property
    def summary_path(self) -> Path:
        return self.artifact_plan.require(ArtifactKind.SUMMARY).path

    @property
    def metadata_path(self) -> Path:
        return self.artifact_plan.require(ArtifactKind.RUN_METADATA).path
