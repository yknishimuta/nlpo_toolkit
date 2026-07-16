from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from nlpo_toolkit.nlp.contracts import NLPBackend

from .execution_session import NLPExecutionSession
from .config_references import ConfigFileReference
from .run_plan import ResolvedAnalysisPlan
from .artifacts.models import ArtifactKind, ArtifactPlan


@dataclass(frozen=True)
class RunContext:
    session: NLPExecutionSession
    sentence_splitter: NLPBackend | None
    artifact_plan: ArtifactPlan


def deduplicate_resolved_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    result: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path not in seen:
            seen.add(path)
            result.append(path)
    return tuple(result)


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    plan: ResolvedAnalysisPlan
    groups_files: Mapping[str, tuple[Path, ...]]
    input_files: tuple[Path, ...]
    cleaned_files: tuple[Path, ...]
    artifact_plan: ArtifactPlan
    config_references: tuple[ConfigFileReference, ...]
    partition_mismatches: tuple[tuple[str, str, int, int], ...] = ()

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
class PartitionRunResult:
    results: tuple[Any, ...]
    summaries: tuple[Mapping[str, object], ...]
    metadata: tuple[Mapping[str, object], ...]
    exit_code: int
    mismatches: tuple[tuple[str, str, int, int], ...] = ()


@dataclass(frozen=True)
class ComparisonRunResult:
    results: tuple[Any, ...]
    metadata: tuple[Mapping[str, object], ...]
