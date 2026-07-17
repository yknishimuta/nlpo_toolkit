from __future__ import annotations

from ..analysis_results import AnalysisResults
from ..publication_models import RunReportPublication
from ..comparison_run_results import ConfiguredComparisonsRunResult
from ..count_context import CountRunContext
from ..partition_run_results import PartitionValidationRunResult
from .environment import collect_runtime_environment
from .metadata import build_run_metadata
from .summary import render_run_summary


def build_run_report(*, context: CountRunContext, analysis: AnalysisResults, partitions: PartitionValidationRunResult, comparisons: ConfiguredComparisonsRunResult) -> RunReportPublication:
    summary = render_run_summary(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
    )
    environment = collect_runtime_environment(context.session.corpus.plan.definition.project_root)
    metadata = build_run_metadata(context=context, analysis=analysis, partitions=partitions, comparisons=comparisons, environment=environment)
    return RunReportPublication(
        artifact_plan=context.artifact_plan,
        summary=summary,
        metadata=metadata,
    )
