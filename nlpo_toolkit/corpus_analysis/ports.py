"""Dependency interfaces and immutable dependency containers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from nlpo_toolkit.nlp.contracts import BuiltNLPBackend
from nlpo_toolkit.cleaner_contracts import (
    CleanerApplicationService,
    CleanerConfigInspection,
)

from .analysis_policy import AnalysisExtractionPolicy
from .analysis_cache_results import AnalysisRecordCacheStatus
from .analysis_records import NLPAnalysisRecord
from .config import AppConfig, NLPConfig
from .publication_ports import CountPublicationDependencies

if TYPE_CHECKING:
    from .archive.contracts import RunArchiveRequest, RunArchiveResult
    from .requests import CorpusPreparationRequest
    from .count_result import CountRunResult
    from .features.models import FunctionWordVocabulary


ConfigLoader = Callable[[Path], AppConfig]
CleanerConfigInspector = Callable[[Path], CleanerConfigInspection]
BackendFactory = Callable[[NLPConfig], BuiltNLPBackend]


class FunctionWordVocabularyLoader(Protocol):
    def __call__(self, path: Path) -> FunctionWordVocabulary: ...


@dataclass(frozen=True)
class AnalysisRecordCacheSettings:
    enabled: bool
    directory: Path
    lock_timeout_sec: float


@dataclass(frozen=True)
class AnalysisRecordRequest:
    text: str
    backend: BuiltNLPBackend
    extraction_policy: AnalysisExtractionPolicy
    cache: AnalysisRecordCacheSettings


@dataclass(frozen=True)
class AnalysisRecordSource:
    records: Iterator[NLPAnalysisRecord]
    cache_status: AnalysisRecordCacheStatus
    cache_key: str


class AnalysisRecordProvider(Protocol):
    def __call__(
        self, request: AnalysisRecordRequest
    ) -> AbstractContextManager[AnalysisRecordSource]: ...


class ArchiveCreator(Protocol):
    def __call__(
        self,
        *,
        run_result: CountRunResult,
        request: RunArchiveRequest,
    ) -> RunArchiveResult: ...


class CountRunner(Protocol):
    def __call__(
        self,
        request: CorpusPreparationRequest,
        *,
        dependencies: RunnerDependencies,
    ) -> CountRunResult: ...


@dataclass(frozen=True)
class CorpusPlanningDependencies:
    load_config: ConfigLoader
    cleaner_inspector: CleanerConfigInspector


@dataclass(frozen=True)
class CorpusPreparationDependencies:
    execute_cleaner: CleanerApplicationService


@dataclass(frozen=True)
class CorpusExecutionDependencies:
    planning: CorpusPlanningDependencies
    preparation: CorpusPreparationDependencies


@dataclass(frozen=True)
class NLPExecutionDependencies:
    backend_factory: BackendFactory
    extraction_policy: AnalysisExtractionPolicy


@dataclass(frozen=True)
class RunnerDependencies:
    corpus: CorpusExecutionDependencies
    nlp: NLPExecutionDependencies
    analysis_records: AnalysisRecordProvider
    publication: CountPublicationDependencies


@dataclass(frozen=True)
class CountCommandDependencies:
    runner: RunnerDependencies
    run_analysis: CountRunner
    archive_creator: ArchiveCreator


@dataclass(frozen=True)
class FeatureCommandDependencies:
    corpus: CorpusExecutionDependencies
    nlp: NLPExecutionDependencies
    load_function_words: FunctionWordVocabularyLoader


@dataclass(frozen=True)
class ConfigNgramDependencies:
    corpus: CorpusExecutionDependencies
