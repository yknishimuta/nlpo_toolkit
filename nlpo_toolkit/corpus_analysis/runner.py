from __future__ import annotations

from . import post_analysis, runtime
from .analysis_orchestration import analyze_corpora
from .ports import RunnerDependencies
from .runner_types import RunResult
from .requests import CorpusPreparationRequest
from .reporting.service import build_run_report
from .result_builder import build_run_result


def run(
    request: CorpusPreparationRequest,
    *,
    dependencies: RunnerDependencies,
) -> RunResult:
    """Core runner. Dependencies are injectable for CLI and tests."""
    context = runtime.prepare_run_context(
        request,
        dependencies=dependencies,
    )
    analysis = analyze_corpora(context, publication=dependencies.publication)
    partitions = post_analysis.execute_partition_validations(
        context=context,
        analysis=analysis,
        publisher=dependencies.publication.partition_artifacts,
    )
    comparisons = post_analysis.execute_group_comparisons(
        context=context,
        analysis=analysis,
        publisher=dependencies.publication.comparison_artifacts,
    )
    report = build_run_report(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
    )
    dependencies.publication.run_report(report)
    return build_run_result(
        context=context,
        analysis=analysis,
        partitions=partitions,
    )
