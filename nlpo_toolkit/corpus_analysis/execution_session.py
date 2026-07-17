"""Shared execution preparation for corpus-backed application commands."""

from __future__ import annotations

from nlpo_toolkit.nlp.roman_numerals import load_roman_exceptions

from .corpus import prepare_corpora
from .ports import CorpusExecutionDependencies, NLPExecutionDependencies
from .planning.build import build_analysis_plan, build_count_plan
from .planning.models import ResolvedAnalysisPlan
from .planning.resolve import prepare_analysis_plan, prepare_count_plan
from .requests import CorpusPreparationRequest
from .session_models import CorpusExecutionSession, NLPExecutionSession


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
