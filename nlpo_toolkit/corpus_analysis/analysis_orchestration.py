from __future__ import annotations

from typing import Mapping

from . import analysis_execution, analysis_outputs
from .analysis_cache.stats import AnalysisCacheRunStats
from .analysis_results import AnalysisResults, GroupAnalysisResult
from .corpus import PreparedCorpus
from .runner_types import RunContext
from .artifacts.models import ArtifactKind

__all__ = ["analyze_corpora"]


def _prepare_counting_text(context: RunContext, corpus: PreparedCorpus) -> str:
    if context.sentence_splitter is None:
        return corpus.prepared_text
    doc = context.sentence_splitter(corpus.prepared_text)
    joined = "\n".join(
        sentence.text or " ".join(token.text for token in sentence.tokens)
        for sentence in doc.sentences
    )
    return joined if joined.strip() else corpus.prepared_text


def _analyze_one_corpus(
    *,
    context: RunContext,
    corpus: PreparedCorpus,
    normalization_map: Mapping[str, str] | None,
    known_words: frozenset[str] | None,
    cache_stats: AnalysisCacheRunStats,
) -> tuple[str, GroupAnalysisResult]:
    artifacts = context.artifact_plan
    token = artifacts.optional(ArtifactKind.TOKEN_ARTIFACT, group=corpus.label)
    token_meta = artifacts.optional(ArtifactKind.TOKEN_ARTIFACT_METADATA,
                                    group=corpus.label)
    trace = artifacts.optional(ArtifactKind.DIAGNOSTIC_TRACE, group=corpus.label)
    record_result = analysis_execution.execute_record_analysis(
        context=context,
        corpus=corpus,
        text=_prepare_counting_text(context, corpus),
        token_artifact_path=token.path if token else None,
        token_artifact_metadata_path=token_meta.path if token_meta else None,
        trace_path=trace.path if trace else None,
        cache_stats=cache_stats,
    )
    output_counter = analysis_outputs.write_group_analysis_outputs(
        plan=context.session.corpus.plan,
        artifact_plan=context.artifact_plan,
        corpus=corpus,
        counter=record_result.counter,
        normalization_map=normalization_map,
        known_words=known_words,
    )
    return (
        corpus.label,
        GroupAnalysisResult(
            files=tuple(corpus.files),
            counter=output_counter.copy(),
            ref_tag_counts=corpus.ref_tag_counts.copy(),
            token_artifact=record_result.token_artifact,
        ),
    )


def analyze_corpora(context: RunContext) -> AnalysisResults:
    plan = context.session.corpus.plan
    normalization_map = analysis_outputs.load_configured_lemma_normalization(plan)
    known_words = analysis_outputs.load_configured_known_words(plan)
    cache_stats = analysis_execution.create_analysis_cache_stats(context)
    groups = (
        _analyze_one_corpus(
            context=context,
            corpus=corpus,
            normalization_map=normalization_map,
            known_words=known_words,
            cache_stats=cache_stats,
        )
        for corpus in context.session.corpus.corpora
    )
    return AnalysisResults.from_groups(
        groups,
        cache_stats=cache_stats,
    )
