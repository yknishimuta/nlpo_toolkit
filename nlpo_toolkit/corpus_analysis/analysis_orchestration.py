from __future__ import annotations

from typing import Mapping

from . import analysis_execution, analysis_outputs
from .analysis_cache import AnalysisCacheRunStats
from .analysis_results import AnalysisResults, GroupAnalysisResult
from .corpus import PreparedCorpus
from .runner_types import RunContext

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
    output_paths: analysis_outputs.GroupOutputPaths,
    normalization_map: Mapping[str, str] | None,
    known_words: frozenset[str] | None,
    cache_stats: AnalysisCacheRunStats,
) -> tuple[str, GroupAnalysisResult]:
    record_result = analysis_execution.execute_record_analysis(
        context=context,
        corpus=corpus,
        text=_prepare_counting_text(context, corpus),
        token_artifact_path=output_paths.token_artifact,
        trace_path=output_paths.trace,
        cache_stats=cache_stats,
    )
    output_result = analysis_outputs.write_group_analysis_outputs(
        plan=context.plan,
        corpus=corpus,
        counter=record_result.counter,
        normalization_map=normalization_map,
        known_words=known_words,
        token_generated_outputs=record_result.generated_outputs,
    )
    return (
        corpus.label,
        GroupAnalysisResult(
            files=tuple(corpus.files),
            counter=output_result.counter.copy(),
            ref_tag_counts=corpus.ref_tag_counts.copy(),
            output_files=output_result.generated_outputs,
            trace_path=output_paths.trace,
            token_artifact=record_result.token_artifact,
        ),
    )


def analyze_corpora(context: RunContext) -> AnalysisResults:
    output_plan = analysis_outputs.build_analysis_output_plan(
        plan=context.plan, corpora=context.prepared_corpora
    )
    normalization_map = analysis_outputs.load_configured_lemma_normalization(context.plan)
    known_words = analysis_outputs.load_configured_known_words(context.plan)
    cache_stats = analysis_execution.create_analysis_cache_stats(context)
    groups = (
        _analyze_one_corpus(
            context=context,
            corpus=corpus,
            output_paths=output_plan.for_group(corpus.label),
            normalization_map=normalization_map,
            known_words=known_words,
            cache_stats=cache_stats,
        )
        for corpus in context.prepared_corpora
    )
    return AnalysisResults.from_groups(
        groups,
        cache_stats=cache_stats,
    )
