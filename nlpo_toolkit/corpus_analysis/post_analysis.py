from __future__ import annotations

from collections import Counter
from nlpo_toolkit.comparison.configured import (
    run_comparisons,
)
from .partition_validation import (
    validate_partitions,
)
from .analysis_results import AnalysisResults
from .artifacts.models import ArtifactKind
from .runner_types import (
    ComparisonRunResult,
    PartitionRunResult,
    PartitionRunMismatch,
    RunContext,
)
from .artifacts.writers.partition import (
    render_partition_json,
    write_partition_csv_artifact,
    write_partition_json_artifact,
)
from .artifacts.writers.comparison import (
    render_comparisons_json,
    write_comparison_csv_artifact,
    write_comparisons_json_artifact,
)


def _group_counters(analysis: AnalysisResults) -> dict[str, Counter[str]]:
    return {label: group.counter for label, group in analysis.groups.items()}


def execute_partition_validations(
    *,
    context: RunContext,
    analysis: AnalysisResults,
) -> PartitionRunResult:
    plan = context.session.corpus.plan
    specs = plan.definition.config.validations.partitions
    partition_results = validate_partitions(specs, _group_counters(analysis))
    exit_code = 0
    mismatches: list[PartitionRunMismatch] = []
    csv_names: dict[str, str] = {}

    for spec, result in zip(specs, partition_results):
        artifact = context.artifact_plan.require(
            ArtifactKind.PARTITION_VALIDATION_CSV, name=spec.name
        )
        csv_names[spec.name] = artifact.path.name
        write_partition_csv_artifact(artifact, result=result)

        if not result.exact_match:
            level = "ERROR" if spec.on_mismatch == "error" else "WARN"
            mismatches.append(PartitionRunMismatch(
                spec.name, level, result.token_delta, result.mismatched_items
            ))
            if spec.on_mismatch == "error":
                exit_code = 1

    if specs:
        partition_json = context.artifact_plan.require(
            ArtifactKind.PARTITION_VALIDATION_JSON
        )
        write_partition_json_artifact(
            partition_json,
            data=render_partition_json(specs, partition_results, csv_names=csv_names),
        )

    return PartitionRunResult(
        validations=tuple(partition_results),
        exit_code=exit_code,
        mismatches=tuple(mismatches),
    )


def execute_group_comparisons(
    *,
    context: RunContext,
    analysis: AnalysisResults,
) -> ComparisonRunResult:
    plan = context.session.corpus.plan
    definition = plan.definition
    comparison_results = run_comparisons(
        specs=definition.config.comparisons,
        counters=_group_counters(analysis),
        analysis_unit=definition.analysis_mode.unit,
    )
    csv_names: dict[str, str] = {}
    for result in comparison_results:
        artifact = context.artifact_plan.require(
            ArtifactKind.GROUP_COMPARISON_CSV, name=result.spec.name
        )
        csv_names[result.spec.name] = artifact.path.name
        write_comparison_csv_artifact(artifact, result=result)

    if comparison_results:
        comparison_json = context.artifact_plan.require(
            ArtifactKind.GROUP_COMPARISONS_JSON
        )
        write_comparisons_json_artifact(
            comparison_json,
            data=render_comparisons_json(comparison_results, csv_names=csv_names),
        )

    return ComparisonRunResult(
        comparisons=tuple(comparison_results),
    )
