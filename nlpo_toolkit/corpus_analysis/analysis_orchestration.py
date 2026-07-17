from __future__ import annotations

from . import analysis_execution
from .analysis_cache.stats import AnalysisCacheRunStats
from .analysis_results import AnalysisResults, GroupAnalysisResult
from .corpus import PreparedCorpus
from .runner_types import RunContext
from .artifacts.models import ArtifactKind
from .artifacts.writers.group import write_group_artifacts
from .postprocessing.service import (
    PostprocessingResources,
    load_postprocessing_resources,
    postprocess_group_counter,
)

__all__ = ["analyze_corpora"]


def _analyze_one_corpus(
    *,
    context: RunContext,
    corpus: PreparedCorpus,
    postprocessing: PostprocessingResources,
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
        text=corpus.prepared_text,
        token_artifact_path=token.path if token else None,
        token_artifact_metadata_path=token_meta.path if token_meta else None,
        trace_path=trace.path if trace else None,
        cache_stats=cache_stats,
    )
    definition = context.session.corpus.plan.definition
    processed = postprocess_group_counter(
        record_result.counter,
        resources=postprocessing,
    )
    write_group_artifacts(
        artifact_plan=context.artifact_plan,
        group=corpus.label,
        counter=processed.counter,
        dictionary=processed.dictionary,
        reference_tag_counts=corpus.ref_tag_counts,
        csv_header=definition.analysis_mode.csv_header,
        reference_tags_enabled=definition.config.ref_tags.enabled,
    )
    return (
        corpus.label,
        GroupAnalysisResult(
            files=tuple(corpus.files),
            counter=processed.counter.copy(),
            ref_tag_counts=corpus.ref_tag_counts.copy(),
            token_artifact=record_result.token_artifact,
        ),
    )


def analyze_corpora(context: RunContext) -> AnalysisResults:
    definition = context.session.corpus.plan.definition
    postprocessing = load_postprocessing_resources(definition)
    cache_stats = analysis_execution.create_analysis_cache_stats(context)
    groups = (
        _analyze_one_corpus(
            context=context,
            corpus=corpus,
            postprocessing=postprocessing,
            cache_stats=cache_stats,
        )
        for corpus in context.session.corpus.corpora
    )
    return AnalysisResults.from_groups(
        groups,
        cache_stats=cache_stats,
    )
