from __future__ import annotations

from collections import Counter
from nlpo_toolkit.comparison.services.configured import compare_configured_counters
from .partition_validation import (
    validate_partitions,
)
from .analysis_results import AnalysisResults
from .publication_models import (
    ComparisonArtifactPublication,
    PartitionArtifactPublication,
)
from .publication_ports import ComparisonArtifactPublisher, PartitionArtifactPublisher
from .comparison_run_results import ConfiguredComparisonsRunResult
from .count_context import CountRunContext
from .partition_run_results import (
    PartitionMismatchSummary,
    PartitionValidationRunResult,
)


def _group_counters(analysis: AnalysisResults) -> dict[str, Counter[str]]:
    return {label: group.counter for label, group in analysis.groups.items()}


def execute_partition_validations(
    *,
    context: CountRunContext,
    analysis: AnalysisResults,
    publisher: PartitionArtifactPublisher,
) -> PartitionValidationRunResult:
    plan = context.session.corpus.plan
    specs = plan.definition.config.validations.partitions
    partition_results = validate_partitions(specs, _group_counters(analysis))
    exit_code = 0
    mismatches: list[PartitionMismatchSummary] = []

    for spec, result in zip(specs, partition_results):
        if not result.exact_match:
            level = "ERROR" if spec.on_mismatch == "error" else "WARN"
            mismatches.append(PartitionMismatchSummary(
                spec.name, level, result.token_delta, result.mismatched_items
            ))
            if spec.on_mismatch == "error":
                exit_code = 1

    publisher(
        PartitionArtifactPublication(
            artifact_plan=context.artifact_plan,
            specs=tuple(specs),
            results=tuple(partition_results),
        )
    )

    return PartitionValidationRunResult(
        validations=tuple(partition_results),
        exit_code=exit_code,
        mismatches=tuple(mismatches),
    )


def execute_group_comparisons(
    *,
    context: CountRunContext,
    analysis: AnalysisResults,
    publisher: ComparisonArtifactPublisher,
) -> ConfiguredComparisonsRunResult:
    plan = context.session.corpus.plan
    definition = plan.definition
    counters = _group_counters(analysis)
    comparison_results = [
        compare_configured_counters(
            counter_a=counters[spec.group_a], counter_b=counters[spec.group_b],
            spec=spec, analysis_unit=definition.analysis_mode.unit,
        )
        for spec in definition.config.comparisons
    ]
    publisher(
        ComparisonArtifactPublication(
            artifact_plan=context.artifact_plan,
            results=tuple(comparison_results),
        )
    )

    return ConfiguredComparisonsRunResult(
        comparisons=tuple(comparison_results),
    )
