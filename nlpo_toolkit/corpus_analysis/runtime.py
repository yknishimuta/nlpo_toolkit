from __future__ import annotations

from nlpo_toolkit.nlp.contracts import NLPBackend

from .execution_session import (
    NLPExecutionSession,
    prepare_count_corpus_session,
    start_nlp_execution_session,
)
from .io_utils import ensure_out_dir
from .ports import RunnerDependencies, SentenceSplitterFactory
from .requests import CorpusPreparationRequest
from .runner_types import RunContext
from .artifacts.planning import build_count_artifact_plan


def initialize_count_sentence_splitter(
    *,
    session: NLPExecutionSession,
    factory: SentenceSplitterFactory | None,
) -> NLPBackend | None:
    if factory is None:
        return None
    return factory(session.corpus.plan.definition.config.nlp)


def prepare_run_context(
    request: CorpusPreparationRequest,
    *,
    dependencies: RunnerDependencies,
) -> RunContext:
    corpus_session = prepare_count_corpus_session(
        request,
        dependencies=dependencies.corpus,
    )
    artifact_plan = build_count_artifact_plan(
        plan=corpus_session.plan, corpora=corpus_session.corpora
    )
    nlp_session = start_nlp_execution_session(
        corpus_session,
        dependencies=dependencies.nlp,
    )
    ensure_out_dir(corpus_session.plan.definition.out_dir)
    sentence_splitter = initialize_count_sentence_splitter(
        session=nlp_session,
        factory=dependencies.count.sentence_splitter_factory,
    )
    return RunContext(
        session=nlp_session,
        sentence_splitter=sentence_splitter,
        artifact_plan=artifact_plan,
    )
