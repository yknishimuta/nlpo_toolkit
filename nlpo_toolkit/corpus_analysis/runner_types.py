from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nlpo_toolkit.nlp.contracts import NLPBackend

from .execution_session import NLPExecutionSession
from .config_references import ConfigFileReference
from .planning.models import ResolvedAnalysisPlan
from .artifacts.models import ArtifactKind, ArtifactPlan
from .partition_validation import PartitionResult
from nlpo_toolkit.comparison.configured import ComparisonResult


@dataclass(frozen=True)
class RunContext:
    session: NLPExecutionSession
    sentence_splitter: NLPBackend | None
    artifact_plan: ArtifactPlan


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    plan: ResolvedAnalysisPlan
    groups_files: Mapping[str, tuple[Path, ...]]
    input_files: tuple[Path, ...]
    cleaned_files: tuple[Path, ...]
    artifact_plan: ArtifactPlan
    config_references: tuple[ConfigFileReference, ...]
    partition_mismatches: tuple[PartitionRunMismatch, ...] = ()

    @property
    def generated_outputs(self) -> tuple[Path, ...]:
        return self.artifact_plan.paths

    @property
    def trace_files(self) -> tuple[Path, ...]:
        return tuple(a.path for a in self.artifact_plan.select(
            kinds={ArtifactKind.DIAGNOSTIC_TRACE}
        ))

    @property
    def output_files(self) -> tuple[Path, ...]:
        return tuple(a.path for a in self.artifact_plan.artifacts
                     if a.kind is not ArtifactKind.DIAGNOSTIC_TRACE)

    @property
    def summary_path(self) -> Path:
        return self.artifact_plan.require(ArtifactKind.SUMMARY).path

    @property
    def metadata_path(self) -> Path:
        return self.artifact_plan.require(ArtifactKind.RUN_METADATA).path


@dataclass(frozen=True)
class PartitionRunMismatch:
    name: str
    level: str
    token_delta: int
    mismatched_items: int


@dataclass(frozen=True)
class PartitionRunResult:
    validations: tuple[PartitionResult, ...]
    exit_code: int
    mismatches: tuple[PartitionRunMismatch, ...] = ()


@dataclass(frozen=True)
class ComparisonRunResult:
    comparisons: tuple[ComparisonResult, ...]
