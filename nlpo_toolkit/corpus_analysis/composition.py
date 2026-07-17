"""Production composition root for corpus-analysis application services."""

from __future__ import annotations

from pathlib import Path
from nlpo_toolkit.backends import create_nlp_backend
from nlpo_toolkit.cleaner_contracts import (
    CleanerConfigInspection,
    CleanerExecutionRequest,
    CleanerExecutionResult,
)
from nlpo_toolkit.nlp.contracts import (
    BuiltNLPBackend,
    NLPBackendSpec,
)

from .analysis_policy import (
    AnalysisExtractionPolicy,
    DEFAULT_ANALYSIS_EXTRACTION_POLICY,
)
from .archive.service import create_run_archive
from .artifacts.publication_adapters import (
    open_record_artifact_session,
    publish_comparison_artifacts,
    publish_group_artifacts,
    publish_partition_artifacts,
)
from .reporting.publication_adapter import publish_run_report
from .config import NLPConfig, load_config
from .ports import (
    BackendFactory,
    ConfigNgramDependencies,
    CorpusExecutionDependencies,
    CorpusPlanningDependencies,
    CorpusPreparationDependencies,
    CountCommandDependencies,
    FeatureCommandDependencies,
    NLPExecutionDependencies,
    RunnerDependencies,
)
from .runner import run
from .publication_ports import CountPublicationDependencies


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


def default_count_publication_dependencies() -> CountPublicationDependencies:
    return CountPublicationDependencies(
        group_artifacts=publish_group_artifacts,
        partition_artifacts=publish_partition_artifacts,
        comparison_artifacts=publish_comparison_artifacts,
        run_report=publish_run_report,
        record_artifacts=open_record_artifact_session,
    )


def default_runner_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        corpus=default_corpus_execution_dependencies(),
        nlp=default_nlp_execution_dependencies(),
        publication=default_count_publication_dependencies(),
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
