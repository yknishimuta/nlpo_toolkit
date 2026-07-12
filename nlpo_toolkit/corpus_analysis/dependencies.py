"""Explicit dependency types and production composition roots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from nlpo_toolkit.backends import BuiltNLPBackend, create_nlp_backend
from nlpo_toolkit.nlp import build_sentence_splitter

from .analysis_policy import (
    AnalysisExtractionPolicy,
    DEFAULT_ANALYSIS_EXTRACTION_POLICY,
)
from .cleaner_runtime import CleanerLoader, load_default_cleaner
from .config import AppConfig, NLPConfig, load_config

if TYPE_CHECKING:
    from .archive import ArchiveOptions, RunArchiveResult
    from .runner_types import RunResult


ConfigLoader = Callable[[Path], AppConfig]
BackendFactory = Callable[[NLPConfig], BuiltNLPBackend]
SentenceSplitterFactory = Callable[[NLPConfig], Any]
ArchiveCreator = Callable[["RunResult", "ArchiveOptions"], "RunArchiveResult"]


class CountRunner(Protocol):
    def __call__(
        self,
        *,
        project_root: Path,
        config_path: Path,
        group_by_file: bool,
        dependencies: RunnerDependencies,
        error_on_empty_group: bool,
        auto_single_cleaned: bool,
    ) -> RunResult: ...


@dataclass(frozen=True)
class CorpusPlanningDependencies:
    load_config: ConfigLoader
    cleaner_loader: CleanerLoader


@dataclass(frozen=True)
class AnalysisDependencies:
    backend_factory: BackendFactory
    extraction_policy: AnalysisExtractionPolicy
    sentence_splitter_factory: SentenceSplitterFactory | None = None


@dataclass(frozen=True)
class RunnerDependencies:
    planning: CorpusPlanningDependencies
    analysis: AnalysisDependencies


@dataclass(frozen=True)
class CountCommandDependencies:
    runner: RunnerDependencies
    run_analysis: CountRunner
    archive_creator: ArchiveCreator


@dataclass(frozen=True)
class FeatureCommandDependencies:
    planning: CorpusPlanningDependencies
    analysis: AnalysisDependencies


@dataclass(frozen=True)
class ConfigNgramDependencies:
    planning: CorpusPlanningDependencies


def _create_sentence_splitter(config: NLPConfig) -> Any:
    return build_sentence_splitter(
        config.language,
        config.stanza_package or "perseus",
        config.cpu_only,
    )


def default_corpus_planning_dependencies() -> CorpusPlanningDependencies:
    return CorpusPlanningDependencies(
        load_config=load_config,
        cleaner_loader=load_default_cleaner,
    )


def default_analysis_dependencies(
    *,
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY,
) -> AnalysisDependencies:
    return AnalysisDependencies(
        backend_factory=lambda config: create_nlp_backend(
            config,
            extraction_policy=extraction_policy,
        ),
        extraction_policy=extraction_policy,
        sentence_splitter_factory=_create_sentence_splitter,
    )


def default_runner_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        planning=default_corpus_planning_dependencies(),
        analysis=default_analysis_dependencies(),
    )


def default_count_command_dependencies() -> CountCommandDependencies:
    from .archive import create_run_archive
    from .runner import run

    return CountCommandDependencies(
        runner=default_runner_dependencies(),
        run_analysis=run,
        archive_creator=create_run_archive,
    )


def default_feature_command_dependencies() -> FeatureCommandDependencies:
    return FeatureCommandDependencies(
        planning=default_corpus_planning_dependencies(),
        analysis=default_analysis_dependencies(),
    )


def default_config_ngram_dependencies() -> ConfigNgramDependencies:
    return ConfigNgramDependencies(
        planning=default_corpus_planning_dependencies(),
    )
