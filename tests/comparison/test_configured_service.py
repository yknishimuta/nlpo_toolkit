from collections import Counter

import pytest

from nlpo_toolkit.comparison.config import ComparisonSortConfig, ComparisonSpec
from nlpo_toolkit.comparison.errors import ComparisonEngineError, ComparisonServiceError
from nlpo_toolkit.comparison.services.configured import compare_configured_counters


def test_configured_result_wraps_engine_result_and_sorts():
    result = compare_configured_counters(
        counter_a=Counter({"b": 2, "a": 2}), counter_b=Counter({"b": 1, "a": 1}),
        spec=ComparisonSpec(name="c", group_a="a", group_b="b",
                            sort=ComparisonSortConfig(by="item", descending=False)),
        analysis_unit="lemma",
    )
    assert result.rows is result.comparison.rows
    assert [row.item for row in result.rows] == ["a", "b"]
    assert result.comparison.scale == 10000
    assert result.group_a_tokens == 4


def test_engine_error_becomes_service_error_with_cause():
    with pytest.raises(ComparisonServiceError) as captured:
        compare_configured_counters(
            counter_a={}, counter_b={"a": 1},
            spec=ComparisonSpec(name="c", group_a="a", group_b="b"),
            analysis_unit="token",
        )
    assert isinstance(captured.value.__cause__, ComparisonEngineError)
