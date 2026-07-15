from __future__ import annotations

from collections import Counter
from dataclasses import fields
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_cache import AnalysisCacheRunStats
from nlpo_toolkit.corpus_analysis.analysis_results import (
    AnalysisResults,
    GroupAnalysisResult,
)


def _stats() -> AnalysisCacheRunStats:
    return AnalysisCacheRunStats(enabled=False, directory="")


def _group(
    *,
    outputs: tuple[Path, ...] = (),
    trace: Path | None = None,
    artifact: dict[str, object] | None = None,
) -> GroupAnalysisResult:
    return GroupAnalysisResult(
        files=(),
        counter=Counter({"rosa": 1}),
        ref_tag_counts=Counter(),
        output_files=outputs,
        trace_path=trace,
        token_artifact=artifact,
    )


def test_groups_mapping_is_immutable_and_detached_from_source() -> None:
    first = _group()
    source = {"a": first}
    results = AnalysisResults.from_groups(source.items(), cache_stats=_stats())
    source["b"] = _group()

    assert tuple(results.groups) == ("a",)
    with pytest.raises(TypeError):
        results.groups["new"] = first  # type: ignore[index]


def test_duplicate_group_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate.*same"):
        AnalysisResults.from_groups(
            (("same", _group()), ("same", _group())),
            cache_stats=_stats(),
        )


def test_derived_outputs_and_artifacts_preserve_group_order(tmp_path: Path) -> None:
    shared = tmp_path / "shared.csv"
    trace = tmp_path / "trace.tsv"
    second = tmp_path / "second.csv"
    results = AnalysisResults.from_groups(
        (
            ("a", _group(outputs=(shared,), trace=trace, artifact={"group": "a"})),
            ("b", _group(outputs=(shared, second), artifact={"group": "b"})),
            ("c", _group()),
        ),
        cache_stats=_stats(),
    )

    assert results.generated_outputs == (
        shared.resolve(),
        trace.resolve(),
        second.resolve(),
    )
    assert results.token_artifact_metadata == ({"group": "a"}, {"group": "b"})
    assert {field.name for field in fields(results)} == {"groups", "cache_stats"}


def test_results_construction_does_not_serialize_cache_stats(monkeypatch) -> None:
    stats = _stats()
    calls = 0

    def record_to_dict():
        nonlocal calls
        calls += 1
        return {"enabled": False}

    monkeypatch.setattr(stats, "to_dict", record_to_dict)
    results = AnalysisResults.from_groups((), cache_stats=stats)

    assert results.cache_stats is stats
    assert calls == 0
