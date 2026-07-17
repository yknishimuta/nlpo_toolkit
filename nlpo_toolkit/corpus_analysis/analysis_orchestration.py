from __future__ import annotations

from . import analysis_execution
from .analysis_cache.stats import AnalysisCacheStatsCollector
from .analysis_results import AnalysisResults, GroupAnalysisResult
from .corpus import PreparedCorpus
from .count_context import CountRunContext
from .publication_models import GroupArtifactPublication
from .publication_ports import CountPublicationDependencies
from .postprocessing.service import (
    PostprocessingResources,
    load_postprocessing_resources,
    postprocess_group_counter,
)

__all__ = ["analyze_corpora"]


def _analyze_one_corpus(
    *,
    context: CountRunContext,
    corpus: PreparedCorpus,
    postprocessing: PostprocessingResources,
    cache_stats: AnalysisCacheStatsCollector,
    publication: CountPublicationDependencies,
) -> tuple[str, GroupAnalysisResult]:
    record_result = analysis_execution.execute_record_analysis(
        context=context,
        corpus=corpus,
        text=corpus.prepared_text,
        publication=publication.record_artifacts,
        cache_stats=cache_stats,
    )
    definition = context.session.corpus.plan.definition
    processed = postprocess_group_counter(
        record_result.counter,
        resources=postprocessing,
    )
    publication.group_artifacts(
        GroupArtifactPublication(
            artifact_plan=context.artifact_plan,
            group=corpus.label,
            counter=processed.counter,
            dictionary=processed.dictionary,
            reference_tag_counts=corpus.ref_tag_counts,
            csv_header=definition.analysis_mode.csv_header,
            reference_tags_enabled=definition.config.ref_tags.enabled,
        )
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


def analyze_corpora(
    context: CountRunContext, *, publication: CountPublicationDependencies
) -> AnalysisResults:
    definition = context.session.corpus.plan.definition
    postprocessing = load_postprocessing_resources(definition)
    cache_stats = analysis_execution.create_analysis_cache_stats(context)
    groups = tuple(
        _analyze_one_corpus(
            context=context,
            corpus=corpus,
            postprocessing=postprocessing,
            cache_stats=cache_stats,
            publication=publication,
        )
        for corpus in context.session.corpus.corpora
    )
    return AnalysisResults.from_groups(
        groups,
        cache_stats=cache_stats.snapshot(),
    )
