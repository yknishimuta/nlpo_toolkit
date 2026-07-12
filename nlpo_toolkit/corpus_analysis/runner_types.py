from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo

from .config import AppConfig, NLPConfig
from .cleaner_runtime import CleanerLoader, CleanerRunner, load_default_cleaner
from .analysis_policy import AnalysisExtractionPolicy, DEFAULT_ANALYSIS_EXTRACTION_POLICY
from .run_plan import RunPlan


ConfigLoader = Callable[[Path], AppConfig]
BackendFactory = Callable[[NLPConfig], BuiltNLPBackend]
SentenceSplitterFactory = Callable[[NLPConfig], Any]


@dataclass(frozen=True)
class RunnerDependencies:
    load_config: ConfigLoader
    backend_factory: BackendFactory
    cleaner: CleanerRunner | None = None
    cleaner_loader: CleanerLoader = load_default_cleaner
    sentence_splitter_factory: SentenceSplitterFactory | None = None
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY


@dataclass(frozen=True)
class RunContext:
    plan: RunPlan
    nlp: Any
    backend_info: NLPBackendInfo
    splitter_nlp: Any | None
    roman_exceptions: frozenset[str]
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY


@dataclass(frozen=True)
class ReferencedConfigFile:
    kind: str
    path: Path
    copy_to_snapshot: bool
    snapshot_path: Path | None = None


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
    plan: RunPlan
    groups_files: Mapping[str, tuple[Path, ...]]
    input_files: tuple[Path, ...]
    cleaned_files: tuple[Path, ...]
    output_files: tuple[Path, ...]
    trace_files: tuple[Path, ...]
    config_files: tuple[ReferencedConfigFile, ...]
    summary_path: Path
    metadata_path: Path

    @property
    def generated_outputs(self) -> tuple[Path, ...]:
        return deduplicate_resolved_paths((*self.output_files, *self.trace_files))


@dataclass(frozen=True)
class GroupAnalysisResult:
    label: str
    files: tuple[Path, ...]
    counter: Counter[str]
    ref_tag_counts: Counter[str]
    generated_outputs: tuple[Path, ...]
    token_artifact: Mapping[str, object] | None = None


@dataclass(frozen=True)
class AnalysisResults:
    groups: tuple[GroupAnalysisResult, ...]
    counters_by_group: Mapping[str, Counter[str]]
    files_by_group: Mapping[str, tuple[Path, ...]]
    ref_tags_by_group: Mapping[str, Counter[str]]
    trace_paths: Mapping[str, Path]
    generated_outputs: tuple[Path, ...]
    token_artifacts: tuple[Mapping[str, object], ...] = ()
    analysis_cache: Mapping[str, object] | None = None


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


@dataclass(frozen=True)
class ComparisonRunResult:
    results: tuple[Any, ...]
    metadata: tuple[Mapping[str, object], ...]
    generated_outputs: tuple[Path, ...]
