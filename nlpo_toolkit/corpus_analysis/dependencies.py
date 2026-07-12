"""Production dependency composition for corpus-analysis commands."""

from __future__ import annotations

from dataclasses import dataclass

from nlpo_toolkit.backends import create_nlp_backend
from nlpo_toolkit.nlp import build_sentence_splitter

from .cleaner_runtime import CleanerLoader, CleanerRunner, load_default_cleaner
from .analysis_policy import AnalysisExtractionPolicy, DEFAULT_ANALYSIS_EXTRACTION_POLICY
from .config import NLPConfig, load_config
from .runner_types import BackendFactory, ConfigLoader, RunnerDependencies


@dataclass(frozen=True)
class FeatureDependencies:
    load_config: ConfigLoader
    backend_factory: BackendFactory
    cleaner: CleanerRunner | None = None
    cleaner_loader: CleanerLoader = load_default_cleaner
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY


@dataclass(frozen=True)
class ConfigNgramDependencies:
    load_config: ConfigLoader
    cleaner: CleanerRunner | None = None
    cleaner_loader: CleanerLoader = load_default_cleaner


def _create_sentence_splitter(config: NLPConfig):
    return build_sentence_splitter(
        config.language,
        config.stanza_package or "perseus",
        config.cpu_only,
    )


def default_runner_dependencies(
    *,
    cleaner: CleanerRunner | None = None,
    cleaner_loader: CleanerLoader = load_default_cleaner,
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY,
) -> RunnerDependencies:
    return RunnerDependencies(
        load_config=load_config,
        backend_factory=lambda config: create_nlp_backend(
            config,
            extraction_policy=extraction_policy,
        ),
        cleaner=cleaner,
        cleaner_loader=cleaner_loader,
        sentence_splitter_factory=_create_sentence_splitter,
        extraction_policy=extraction_policy,
    )


def default_feature_dependencies(
    *,
    cleaner: CleanerRunner | None = None,
    cleaner_loader: CleanerLoader = load_default_cleaner,
    extraction_policy: AnalysisExtractionPolicy = DEFAULT_ANALYSIS_EXTRACTION_POLICY,
) -> FeatureDependencies:
    return FeatureDependencies(
        load_config=load_config,
        backend_factory=lambda config: create_nlp_backend(
            config,
            extraction_policy=extraction_policy,
        ),
        cleaner=cleaner,
        cleaner_loader=cleaner_loader,
        extraction_policy=extraction_policy,
    )


def default_config_ngram_dependencies(
    *,
    cleaner: CleanerRunner | None = None,
    cleaner_loader: CleanerLoader = load_default_cleaner,
) -> ConfigNgramDependencies:
    return ConfigNgramDependencies(
        load_config=load_config,
        cleaner=cleaner,
        cleaner_loader=cleaner_loader,
    )
