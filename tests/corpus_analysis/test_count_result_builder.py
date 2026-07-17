from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nlpo_toolkit.corpus_analysis.artifacts.models import ArtifactPlan
from nlpo_toolkit.corpus_analysis.count_result_builder import build_count_run_result
from nlpo_toolkit.corpus_analysis.partition_run_results import (
    PartitionMismatchSummary,
    PartitionValidationRunResult,
)


def _context(plan, artifact_plan: ArtifactPlan):
    return SimpleNamespace(
        session=SimpleNamespace(corpus=SimpleNamespace(plan=plan)),
        artifact_plan=artifact_plan,
    )


def test_builder_uses_only_analyzed_files_and_preserves_order_and_state(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    references = (object(),)
    definition = SimpleNamespace(
        cleaner_plan=None,
        config_files=SimpleNamespace(references=references),
    )
    plan = SimpleNamespace(definition=definition, cleaned_dir=None)
    artifacts = ArtifactPlan(())
    analysis = SimpleNamespace(
        groups={
            "g": SimpleNamespace(files=(first, first, second)),
            "unused-empty": SimpleNamespace(files=()),
        }
    )
    mismatch = PartitionMismatchSummary("p", "WARN", 1, 2)
    partitions = PartitionValidationRunResult((), 4, (mismatch,))

    result = build_count_run_result(
        context=_context(plan, artifacts),  # type: ignore[arg-type]
        analysis=analysis,  # type: ignore[arg-type]
        partitions=partitions,
    )

    assert result.groups_files == {
        "g": (first.resolve(), second.resolve()),
        "unused-empty": (),
    }
    assert result.input_files == (first.resolve(), second.resolve())
    assert result.cleaned_files == ()
    assert result.config_references is references
    assert result.exit_code == 4
    assert result.partition_mismatches == (mismatch,)
    assert result.artifact_plan is artifacts


def test_builder_classifies_only_used_files_below_cleaned_root(tmp_path: Path) -> None:
    cleaned_root = (tmp_path / "cleaned").resolve()
    cleaned = cleaned_root / "used.txt"
    outside = (tmp_path / "elsewhere.txt").resolve()
    source = (tmp_path / "source.txt").resolve()
    definition = SimpleNamespace(
        cleaner_plan=SimpleNamespace(
            inspection=SimpleNamespace(input_files=(source, source))
        ),
        config_files=SimpleNamespace(references=()),
    )
    plan = SimpleNamespace(definition=definition, cleaned_dir=cleaned_root)
    analysis = SimpleNamespace(
        groups={"g": SimpleNamespace(files=(cleaned, outside, cleaned))}
    )

    result = build_count_run_result(
        context=_context(plan, ArtifactPlan(())),  # type: ignore[arg-type]
        analysis=analysis,  # type: ignore[arg-type]
        partitions=PartitionValidationRunResult((), 0),
    )

    assert result.input_files == (source,)
    assert result.cleaned_files == (cleaned,)

