"""Production dependency composition for corpus-analysis commands."""

from __future__ import annotations

from dataclasses import dataclass

from nlpo_toolkit.backends import create_nlp_backend
from nlpo_toolkit.nlp import build_sentence_splitter

from .cleaner_runtime import CleanerLoader, CleanerRunner, load_default_cleaner
from .config import NLPConfig, load_config
from .runner_types import BackendFactory, ConfigLoader, RunnerDependencies


@dataclass(frozen=True)
class FeatureDependencies:
    load_config: ConfigLoader
    backend_factory: BackendFactory
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
) -> RunnerDependencies:
    return RunnerDependencies(
        load_config=load_config,
        backend_factory=create_nlp_backend,
        cleaner=cleaner,
        cleaner_loader=cleaner_loader,
        sentence_splitter_factory=_create_sentence_splitter,
    )


def default_feature_dependencies(
    *,
    cleaner: CleanerRunner | None = None,
    cleaner_loader: CleanerLoader = load_default_cleaner,
) -> FeatureDependencies:
    return FeatureDependencies(
        load_config=load_config,
        backend_factory=create_nlp_backend,
        cleaner=cleaner,
        cleaner_loader=cleaner_loader,
    )
