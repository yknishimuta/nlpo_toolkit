from __future__ import annotations

from collections import Counter
from dataclasses import fields
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_cache.stats import AnalysisCacheRunStats
from nlpo_toolkit.corpus_analysis.analysis_results import (
    AnalysisResults,
    GroupAnalysisResult,
)


def _stats() -> AnalysisCacheRunStats:
    return AnalysisCacheRunStats(enabled=False, directory="")


def _group(
    *,
    artifact: dict[str, object] | None = None,
) -> GroupAnalysisResult:
    return GroupAnalysisResult(
        files=(),
        counter=Counter({"rosa": 1}),
        ref_tag_counts=Counter(),
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


def test_artifact_metadata_preserves_group_order() -> None:
    results = AnalysisResults.from_groups(
        (
            ("a", _group(artifact={"group": "a"})),
            ("b", _group(artifact={"group": "b"})),
            ("c", _group()),
        ),
        cache_stats=_stats(),
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
