from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Tuple

from nlpo_toolkit.backends import BuiltNLPBackend, NLPBackendInfo

from .config import AppConfig
from .run_plan import RunPlan


@dataclass(frozen=True)
class RunnerDependencies:
    load_config: Callable[[Path], AppConfig | Mapping[str, object]]
    clean_module: Any
    render_stanza_package_table: Callable[..., List[str]]
    build_pipeline: Callable[[str, str, bool], Tuple[Any, str]] | None = None
    backend_factory: Callable[[Any], BuiltNLPBackend] | None = None
    build_sentence_splitter: Callable[..., Any] | None = None


@dataclass(frozen=True)
class RunContext:
    plan: RunPlan
    nlp: Any
    backend_info: NLPBackendInfo
    stanza_package: Any
    splitter_nlp: Any | None
    roman_exceptions: frozenset[str]


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
