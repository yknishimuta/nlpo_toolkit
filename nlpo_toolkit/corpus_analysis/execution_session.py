"""Shared execution preparation for corpus-backed application commands."""

from __future__ import annotations

from dataclasses import dataclass

from nlpo_toolkit.nlp.contracts import BuiltNLPBackend
from nlpo_toolkit.nlp.roman_numerals import load_roman_exceptions

from .analysis_policy import AnalysisExtractionPolicy
from .corpus import PreparedCorpus, prepare_corpora
from .ports import CorpusExecutionDependencies, NLPExecutionDependencies
from .preprocessing import prepare_analysis_plan
from .requests import CorpusPreparationRequest
from .run_plan import (
    ResolvedAnalysisPlan,
    build_analysis_plan,
    build_count_plan,
    prepare_count_plan,
)


@dataclass(frozen=True)
class CorpusExecutionSession:
    plan: ResolvedAnalysisPlan
    corpora: tuple[PreparedCorpus, ...]


@dataclass(frozen=True)
class NLPExecutionSession:
    corpus: CorpusExecutionSession
    backend: BuiltNLPBackend
    extraction_policy: AnalysisExtractionPolicy
    roman_exceptions: frozenset[str]


def _build_corpus_execution_session(
    plan: ResolvedAnalysisPlan,
) -> CorpusExecutionSession:
    corpora = prepare_corpora(
        work_items=plan.work_items,
        config=plan.definition.config,
        config_files=plan.definition.config_files,
    )
    return CorpusExecutionSession(plan=plan, corpora=tuple(corpora))


def prepare_analysis_corpus_session(
    request: CorpusPreparationRequest,
    *,
    dependencies: CorpusExecutionDependencies,
) -> CorpusExecutionSession:
    definition = build_analysis_plan(request, dependencies=dependencies.planning)
    resolved = prepare_analysis_plan(
        definition,
        dependencies=dependencies.preparation,
    )
    return _build_corpus_execution_session(resolved)


def prepare_count_corpus_session(
    request: CorpusPreparationRequest,
    *,
    dependencies: CorpusExecutionDependencies,
) -> CorpusExecutionSession:
    definition = build_count_plan(request, dependencies=dependencies.planning)
    resolved = prepare_count_plan(
        definition,
        dependencies=dependencies.preparation,
    )
    return _build_corpus_execution_session(resolved)


def start_nlp_execution_session(
    corpus: CorpusExecutionSession,
    *,
    dependencies: NLPExecutionDependencies,
) -> NLPExecutionSession:
    definition = corpus.plan.definition
    backend = dependencies.backend_factory(definition.config.nlp)
    roman_exceptions_path = definition.config_files.path(
        "filters.roman_exceptions_file"
    )
    roman_exceptions = (
        load_roman_exceptions(roman_exceptions_path)
        if roman_exceptions_path is not None
        else frozenset()
    )
    return NLPExecutionSession(
        corpus=corpus,
        backend=backend,
        extraction_policy=dependencies.extraction_policy,
        roman_exceptions=roman_exceptions,
    )
