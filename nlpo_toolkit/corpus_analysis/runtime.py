from __future__ import annotations

from .execution_session import (
    prepare_count_corpus_session,
    start_nlp_execution_session,
)
from .io_utils import ensure_out_dir
from .ports import RunnerDependencies
from .requests import CorpusPreparationRequest
from .count_context import CountRunContext
from .artifacts.planning import build_count_artifact_plan


def prepare_run_context(
    request: CorpusPreparationRequest,
    *,
    dependencies: RunnerDependencies,
) -> CountRunContext:
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
    return CountRunContext(
        session=nlp_session,
        artifact_plan=artifact_plan,
    )
