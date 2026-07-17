from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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

