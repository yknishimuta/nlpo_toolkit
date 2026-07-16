from __future__ import annotations

from ..analysis_results import AnalysisResults
from ..artifacts.models import ArtifactKind
from ..artifacts.writers.report import write_run_metadata_artifact, write_summary_artifact
from ..runner_types import ComparisonRunResult, PartitionRunResult, RunContext
from .environment import collect_runtime_environment
from .metadata import build_run_metadata, run_metadata_to_json_value
from .summary import render_run_summary


def write_run_report(*, context: RunContext, analysis: AnalysisResults, partitions: PartitionRunResult, comparisons: ComparisonRunResult) -> None:
    write_summary_artifact(
        context.artifact_plan.require(ArtifactKind.SUMMARY),
        content=render_run_summary(context=context, analysis=analysis, partitions=partitions, comparisons=comparisons),
    )
    environment = collect_runtime_environment(context.session.corpus.plan.definition.project_root)
    metadata = build_run_metadata(context=context, analysis=analysis, partitions=partitions, comparisons=comparisons, environment=environment)
    write_run_metadata_artifact(
        context.artifact_plan.require(ArtifactKind.RUN_METADATA),
        data=run_metadata_to_json_value(metadata),
    )
