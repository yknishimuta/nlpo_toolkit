from __future__ import annotations

from pathlib import Path
from typing import Mapping

from nlpo_toolkit.comparison.configured import (
    run_comparisons,
)
from nlpo_toolkit.comparison.writers import (
    comparison_csv_name,
    comparison_result_meta,
    write_group_comparison_csv,
    write_group_comparisons_json,
)
from .partition_validation import (
    partition_result_meta,
    partition_result_summary,
    sanitize_partition_name,
    validate_partitions,
    write_partition_validation_csv,
    write_partition_validation_json,
)
from .runner_types import (
    AnalysisResults,
    ComparisonRunResult,
    PartitionRunResult,
    RunContext,
)


def execute_partition_validations(
    *,
    context: RunContext,
    analysis: AnalysisResults,
) -> PartitionRunResult:
    plan = context.plan
    partition_results = validate_partitions(plan.partition_specs, analysis.counters_by_group)
    summaries: list[Mapping[str, object]] = []
    metadata: list[Mapping[str, object]] = []
    generated_outputs: list[Path] = []
    exit_code = 0
    mismatches: list[tuple[str, str, int, int]] = []

    for spec, result in zip(plan.partition_specs, partition_results):
        csv_name = f"partition_validation_{sanitize_partition_name(spec.name)}.csv"
        csv_path = plan.out_dir / csv_name
        write_partition_validation_csv(csv_path, result)
        generated_outputs.append(csv_path)
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
        partition_json_path = plan.out_dir / "partition_validation.json"
        write_partition_validation_json(partition_json_path, summaries)
        generated_outputs.append(partition_json_path)

    return PartitionRunResult(
        results=tuple(partition_results),
        summaries=tuple(summaries),
        metadata=tuple(metadata),
        generated_outputs=tuple(generated_outputs),
        exit_code=exit_code,
        mismatches=tuple(mismatches),
    )


def execute_group_comparisons(
    *,
    context: RunContext,
    analysis: AnalysisResults,
) -> ComparisonRunResult:
    plan = context.plan
    comparison_results = run_comparisons(
        specs=plan.comparison_specs,
        counters=analysis.counters_by_group,
        analysis_unit=plan.analysis_unit,
    )
    generated_outputs: list[Path] = []
    metadata: list[Mapping[str, object]] = []
    for result in comparison_results:
        csv_name = comparison_csv_name(result.spec)
        csv_path = plan.out_dir / csv_name
        write_group_comparison_csv(csv_path, result)
        generated_outputs.append(csv_path)
        metadata.append(comparison_result_meta(result, csv_name=csv_name))

    if comparison_results:
        comparison_json_path = plan.out_dir / "group_comparisons.json"
        write_group_comparisons_json(comparison_json_path, comparison_results)
        generated_outputs.append(comparison_json_path)

    return ComparisonRunResult(
        results=tuple(comparison_results),
        metadata=tuple(metadata),
        generated_outputs=tuple(generated_outputs),
    )
