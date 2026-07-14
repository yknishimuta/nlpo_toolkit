"""Production composition root for corpus-analysis application services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nlpo_toolkit.backends import BuiltNLPBackend, create_nlp_backend
from nlpo_toolkit.backends.stanza_backend import StanzaBackend
from nlpo_toolkit.cleaner_contracts import CleanerConfigInspection

from .analysis_policy import (
    AnalysisExtractionPolicy,
    DEFAULT_ANALYSIS_EXTRACTION_POLICY,
)
from .archive import create_run_archive
from .cleaner_runtime import load_default_cleaner
from .config import NLPConfig, load_config
from .ports import (
    AnalysisDependencies,
    BackendFactory,
    ConfigNgramDependencies,
    CorpusPlanningDependencies,
    CountCommandDependencies,
    FeatureCommandDependencies,
    RunnerDependencies,
)
from .runner import run


def _inspect_cleaner_config(path: Path) -> CleanerConfigInspection:
    # Keep the optional bundled cleaner package lazy until preprocessing is inspected.
    from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config

    return inspect_cleaner_config(path)


def _create_sentence_splitter(config: NLPConfig) -> Any:
    return StanzaBackend(
        lang=config.language,
        package=config.stanza_package or "perseus",
        use_gpu=not config.cpu_only,
        processors="tokenize",
    )


def _backend_factory_for(
    extraction_policy: AnalysisExtractionPolicy,
) -> BackendFactory:
    def create_backend(config: NLPConfig) -> BuiltNLPBackend:
        return create_nlp_backend(
            config,
            extraction_policy=extraction_policy,
        )

    return create_backend


def default_corpus_planning_dependencies() -> CorpusPlanningDependencies:
    return CorpusPlanningDependencies(
        load_config=load_config,
        cleaner_loader=load_default_cleaner,
        cleaner_inspector=_inspect_cleaner_config,
    )


def default_analysis_dependencies(
    *,
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY,
) -> AnalysisDependencies:
    return AnalysisDependencies(
        backend_factory=_backend_factory_for(extraction_policy),
        extraction_policy=extraction_policy,
        sentence_splitter_factory=_create_sentence_splitter,
    )


def default_runner_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        planning=default_corpus_planning_dependencies(),
        analysis=default_analysis_dependencies(),
    )


def default_count_command_dependencies() -> CountCommandDependencies:
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
