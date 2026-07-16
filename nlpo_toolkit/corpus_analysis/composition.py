"""Production composition root for corpus-analysis application services."""

from __future__ import annotations

from pathlib import Path
from nlpo_toolkit.backends import create_nlp_backend
from nlpo_toolkit.backends.stanza_backend import StanzaBackend
from nlpo_toolkit.cleaner_contracts import (
    CleanerConfigInspection,
    CleanerExecutionRequest,
    CleanerExecutionResult,
)
from nlpo_toolkit.nlp.contracts import (
    BuiltNLPBackend,
    NLPBackend,
    NLPBackendSpec,
)

from .analysis_policy import (
    AnalysisExtractionPolicy,
    DEFAULT_ANALYSIS_EXTRACTION_POLICY,
)
from .archive.service import create_run_archive
from .config import NLPConfig, load_config
from .ports import (
    BackendFactory,
    ConfigNgramDependencies,
    CorpusExecutionDependencies,
    CorpusPlanningDependencies,
    CorpusPreparationDependencies,
    CountRuntimeDependencies,
    CountCommandDependencies,
    FeatureCommandDependencies,
    NLPExecutionDependencies,
    RunnerDependencies,
)
from .runner import run


def _inspect_cleaner_config(path: Path) -> CleanerConfigInspection:
    # Keep the optional bundled cleaner package lazy until preprocessing is inspected.
    from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config

    return inspect_cleaner_config(path)


def _execute_cleaner(request: CleanerExecutionRequest) -> CleanerExecutionResult:
    from nlpo_toolkit.latin.cleaners.service import execute_cleaner

    return execute_cleaner(request)


def _to_backend_spec(config: NLPConfig) -> NLPBackendSpec:
    return NLPBackendSpec(
        backend=config.backend,
        language=config.language,
        stanza_package=config.stanza_package,
        model_name=config.model_name,
        use_gpu=not config.cpu_only,
    )


def _create_sentence_splitter(config: NLPConfig) -> NLPBackend:
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
            _to_backend_spec(config),
            processors=extraction_policy.processors,
        )

    return create_backend


def default_corpus_planning_dependencies() -> CorpusPlanningDependencies:
    return CorpusPlanningDependencies(
        load_config=load_config,
        cleaner_inspector=_inspect_cleaner_config,
    )


def default_corpus_preparation_dependencies() -> CorpusPreparationDependencies:
    return CorpusPreparationDependencies(execute_cleaner=_execute_cleaner)


def default_corpus_execution_dependencies() -> CorpusExecutionDependencies:
    return CorpusExecutionDependencies(
        planning=default_corpus_planning_dependencies(),
        preparation=default_corpus_preparation_dependencies(),
    )


def default_nlp_execution_dependencies(
    *,
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY,
) -> NLPExecutionDependencies:
    return NLPExecutionDependencies(
        backend_factory=_backend_factory_for(extraction_policy),
        extraction_policy=extraction_policy,
    )


def default_count_runtime_dependencies() -> CountRuntimeDependencies:
    return CountRuntimeDependencies(sentence_splitter_factory=_create_sentence_splitter)


def default_runner_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        corpus=default_corpus_execution_dependencies(),
        nlp=default_nlp_execution_dependencies(),
        count=default_count_runtime_dependencies(),
    )


def default_count_command_dependencies() -> CountCommandDependencies:
    return CountCommandDependencies(
        runner=default_runner_dependencies(),
        run_analysis=run,
        archive_creator=create_run_archive,
    )


def default_feature_command_dependencies() -> FeatureCommandDependencies:
    return FeatureCommandDependencies(
        corpus=default_corpus_execution_dependencies(),
        nlp=default_nlp_execution_dependencies(),
    )


def default_config_ngram_dependencies() -> ConfigNgramDependencies:
    return ConfigNgramDependencies(
        corpus=default_corpus_execution_dependencies(),
    )
