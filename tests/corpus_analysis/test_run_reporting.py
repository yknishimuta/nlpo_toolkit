from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.corpus_analysis.run_reporting import merge_generated_outputs


def test_merge_generated_outputs_deduplicates_without_reordering(tmp_path: Path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"

    assert merge_generated_outputs((first, second), (first,), (second,)) == (
        first,
        second,
    )


def test_run_reporting_functions_have_canonical_module() -> None:
    from nlpo_toolkit.corpus_analysis.run_reporting import build_final_run_metadata

    assert build_final_run_metadata.__module__.endswith(".run_reporting")
