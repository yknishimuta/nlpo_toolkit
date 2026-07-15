"""Dependency interfaces and immutable dependency containers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from nlpo_toolkit.nlp.contracts import BuiltNLPBackend, NLPBackend
from nlpo_toolkit.cleaner_contracts import (
    CleanerConfigInspection,
    CleanerLoader,
)

from .analysis_policy import AnalysisExtractionPolicy
from .config import AppConfig, NLPConfig

if TYPE_CHECKING:
    from .archive_types import RunArchiveRequest, RunArchiveResult
    from .requests import CorpusPreparationRequest
    from .runner_types import RunResult


ConfigLoader = Callable[[Path], AppConfig]
CleanerConfigInspector = Callable[[Path], CleanerConfigInspection]
BackendFactory = Callable[[NLPConfig], BuiltNLPBackend]
SentenceSplitterFactory = Callable[[NLPConfig], NLPBackend]


class ArchiveCreator(Protocol):
    def __call__(
        self,
        *,
        run_result: RunResult,
        request: RunArchiveRequest,
    ) -> RunArchiveResult: ...


class CountRunner(Protocol):
    def __call__(
        self,
        request: CorpusPreparationRequest,
        *,
        dependencies: RunnerDependencies,
    ) -> RunResult: ...


@dataclass(frozen=True)
class CorpusPlanningDependencies:
    load_config: ConfigLoader
    cleaner_inspector: CleanerConfigInspector


@dataclass(frozen=True)
class CorpusPreparationDependencies:
    cleaner_loader: CleanerLoader


@dataclass(frozen=True)
class AnalysisDependencies:
    backend_factory: BackendFactory
    extraction_policy: AnalysisExtractionPolicy
    sentence_splitter_factory: SentenceSplitterFactory | None = None


@dataclass(frozen=True)
class RunnerDependencies:
    planning: CorpusPlanningDependencies
    preparation: CorpusPreparationDependencies
    analysis: AnalysisDependencies


@dataclass(frozen=True)
class CountCommandDependencies:
    runner: RunnerDependencies
    run_analysis: CountRunner
    archive_creator: ArchiveCreator


@dataclass(frozen=True)
class FeatureCommandDependencies:
    planning: CorpusPlanningDependencies
    preparation: CorpusPreparationDependencies
    analysis: AnalysisDependencies


@dataclass(frozen=True)
class ConfigNgramDependencies:
    planning: CorpusPlanningDependencies
    preparation: CorpusPreparationDependencies
