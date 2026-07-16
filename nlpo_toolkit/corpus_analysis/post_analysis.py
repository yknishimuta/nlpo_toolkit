from __future__ import annotations

from collections import Counter
from typing import Mapping

from nlpo_toolkit.comparison.configured import (
    run_comparisons,
)
from nlpo_toolkit.comparison.writers import (
    comparison_result_meta,
    write_group_comparison_csv,
    write_group_comparisons_json,
)
from .partition_validation import (
    partition_result_meta,
    partition_result_summary,
    validate_partitions,
    write_partition_validation_csv,
    write_partition_validation_json,
)
from .analysis_results import AnalysisResults
from .artifacts.models import ArtifactKind
from .runner_types import (
    ComparisonRunResult,
    PartitionRunResult,
    RunContext,
)


def _group_counters(analysis: AnalysisResults) -> dict[str, Counter[str]]:
    return {label: group.counter for label, group in analysis.groups.items()}


def execute_partition_validations(
    *,
    context: RunContext,
    analysis: AnalysisResults,
) -> PartitionRunResult:
    plan = context.session.corpus.plan
    partition_results = validate_partitions(plan.partition_specs, _group_counters(analysis))
    summaries: list[Mapping[str, object]] = []
    metadata: list[Mapping[str, object]] = []
    exit_code = 0
    mismatches: list[tuple[str, str, int, int]] = []

    for spec, result in zip(plan.partition_specs, partition_results):
        csv_path = context.artifact_plan.require(
            ArtifactKind.PARTITION_VALIDATION_CSV, name=spec.name
        ).path
        csv_name = csv_path.name
        write_partition_validation_csv(csv_path, result)
        summaries.append(partition_result_summary(spec, result, csv_name=csv_name))
        metadata.append(partition_result_meta(spec, result))

        if not result.exact_match:
            level = "ERROR" if spec.on_mismatch == "error" else "WARN"
            mismatches.append(
                (spec.name, level, result.token_delta, result.mismatched_items)
            )
            if spec.on_mismatch == "error":
                exit_code = 1

    if plan.partition_specs:
        partition_json_path = context.artifact_plan.require(
            ArtifactKind.PARTITION_VALIDATION_JSON
        ).path
        write_partition_validation_json(partition_json_path, summaries)

    return PartitionRunResult(
        results=tuple(partition_results),
        summaries=tuple(summaries),
        metadata=tuple(metadata),
        exit_code=exit_code,
        mismatches=tuple(mismatches),
    )


def execute_group_comparisons(
    *,
    context: RunContext,
    analysis: AnalysisResults,
) -> ComparisonRunResult:
    plan = context.session.corpus.plan
    comparison_results = run_comparisons(
        specs=plan.comparison_specs,
        counters=_group_counters(analysis),
        analysis_unit=plan.analysis_unit,
    )
    metadata: list[Mapping[str, object]] = []
    for result in comparison_results:
        csv_path = context.artifact_plan.require(
            ArtifactKind.GROUP_COMPARISON_CSV, name=result.spec.name
        ).path
        csv_name = csv_path.name
        write_group_comparison_csv(csv_path, result)
        metadata.append(comparison_result_meta(result, csv_name=csv_name))

    if comparison_results:
        comparison_json_path = context.artifact_plan.require(
            ArtifactKind.GROUP_COMPARISONS_JSON
        ).path
        csv_names = {
            result.spec.name: context.artifact_plan.require(
                ArtifactKind.GROUP_COMPARISON_CSV, name=result.spec.name
            ).path.name
            for result in comparison_results
        }
        write_group_comparisons_json(
            comparison_json_path, comparison_results, csv_names=csv_names
        )

    return ComparisonRunResult(
        results=tuple(comparison_results),
        metadata=tuple(metadata),
    )
