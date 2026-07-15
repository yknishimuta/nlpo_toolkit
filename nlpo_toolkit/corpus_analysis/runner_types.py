from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from nlpo_toolkit.nlp.contracts import NLPBackend

from .execution_session import NLPExecutionSession
from .config_references import ConfigFileReference
from .run_plan import ResolvedAnalysisPlan


@dataclass(frozen=True)
class RunContext:
    session: NLPExecutionSession
    sentence_splitter: NLPBackend | None


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
    output_files: tuple[Path, ...]
    trace_files: tuple[Path, ...]
    config_references: tuple[ConfigFileReference, ...]
    summary_path: Path
    metadata_path: Path
    partition_mismatches: tuple[tuple[str, str, int, int], ...] = ()

    @property
    def generated_outputs(self) -> tuple[Path, ...]:
        return deduplicate_resolved_paths((*self.output_files, *self.trace_files))


@dataclass(frozen=True)
class DictCheckOutput:
    known: Counter[str]
    unknown: Counter[str]
    generated_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class PartitionRunResult:
    results: tuple[Any, ...]
    summaries: tuple[Mapping[str, object], ...]
    metadata: tuple[Mapping[str, object], ...]
    generated_outputs: tuple[Path, ...]
    exit_code: int
    mismatches: tuple[tuple[str, str, int, int], ...] = ()


@dataclass(frozen=True)
class ComparisonRunResult:
    results: tuple[Any, ...]
    metadata: tuple[Mapping[str, object], ...]
    generated_outputs: tuple[Path, ...]
