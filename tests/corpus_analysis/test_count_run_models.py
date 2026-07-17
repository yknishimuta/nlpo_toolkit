from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.corpus_analysis.artifacts.models import (
    ArtifactKind,
    ArtifactPlan,
    PlannedArtifact,
)
from nlpo_toolkit.corpus_analysis.comparison_run_results import (
    ConfiguredComparisonsRunResult,
)
from nlpo_toolkit.corpus_analysis.count_result import CountRunResult
from nlpo_toolkit.corpus_analysis.partition_run_results import (
    PartitionMismatchSummary,
    PartitionValidationRunResult,
)


def test_count_result_derives_all_output_views_from_artifact_plan(
    tmp_path: Path,
) -> None:
    frequency = PlannedArtifact(
        ArtifactKind.FREQUENCY, tmp_path / "frequency.csv", group="g"
    )
    trace = PlannedArtifact(
        ArtifactKind.DIAGNOSTIC_TRACE, tmp_path / "trace.tsv", group="g"
    )
    summary = PlannedArtifact(ArtifactKind.SUMMARY, tmp_path / "summary.txt")
    metadata = PlannedArtifact(ArtifactKind.RUN_METADATA, tmp_path / "run_meta.json")
    plan = ArtifactPlan((frequency, trace, summary, metadata))
    result = CountRunResult(
        exit_code=0,
        plan=object(),  # type: ignore[arg-type]
        groups_files={},
        input_files=(),
        cleaned_files=(),
        artifact_plan=plan,
        config_references=(),
    )

    assert result.generated_outputs == plan.paths
    assert result.trace_files == (trace.path,)
    assert result.output_files == (frequency.path, summary.path, metadata.path)
    assert result.summary_path == summary.path
    assert result.metadata_path == metadata.path


def test_partition_and_comparison_run_results_have_semantic_defaults() -> None:
    partition = PartitionValidationRunResult(validations=(), exit_code=3)
    mismatch = PartitionMismatchSummary("p", "ERROR", 2, 1)
    comparisons = (object(), object())

    assert partition.validations == ()
    assert partition.exit_code == 3
    assert partition.mismatches == ()
    assert (mismatch.name, mismatch.level, mismatch.token_delta, mismatch.mismatched_items) == (
        "p", "ERROR", 2, 1
    )
    assert ConfiguredComparisonsRunResult().comparisons == ()
    assert ConfiguredComparisonsRunResult(  # type: ignore[arg-type]
        comparisons
    ).comparisons == comparisons

