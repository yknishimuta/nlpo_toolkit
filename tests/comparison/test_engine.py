from __future__ import annotations

import math

import pytest

from nlpo_toolkit.comparison.engine import (
    FrequencyTable,
    PairwiseComparisonOptions,
    ZeroHandling,
    ZeroHandlingMode,
    compare_many,
    compare_pair,
)
from nlpo_toolkit.comparison.errors import ComparisonEngineError
from nlpo_toolkit.comparison.metrics import normalized_rate


def test_frequency_table_validates_counts_and_freezes_input() -> None:
    counts = {"item_a": 2, "item_b": 1.5}
    table = FrequencyTable.from_counts("table_a", counts)
    counts["item_a"] = 99

    assert table.label == "table_a"
    assert table.counts["item_a"] == 2.0
    assert table.total == pytest.approx(3.5)


@pytest.mark.parametrize(
    "counts",
    [
        {"item_a": -1},
        {"item_a": math.nan},
        {"item_a": math.inf},
        {"item_a": True},
        {"item_a": 0},
    ],
)
def test_frequency_table_rejects_invalid_counts(counts: dict[str, object]) -> None:
    with pytest.raises(ComparisonEngineError):
        FrequencyTable.from_counts("table_a", counts)  # type: ignore[arg-type]


def test_normalized_rate() -> None:
    assert normalized_rate(10, 100, scale=10000) == pytest.approx(1000)


def test_pairwise_equal_distribution() -> None:
    result = compare_pair(
        FrequencyTable.from_counts("group_a", {"item": 10, "other": 90}),
        FrequencyTable.from_counts("group_b", {"item": 20, "other": 180}),
        options=PairwiseComparisonOptions(
            scale=10000,
            zero_handling=ZeroHandling(ZeroHandlingMode.ZERO_ONLY, 0.5),
        ),
    )
    row = {row.item: row for row in result.rows}["item"]

    assert row.rate_difference == pytest.approx(0)
    assert row.log_ratio == pytest.approx(0)
    assert row.log_likelihood == pytest.approx(0)
    assert row.direction == "equal"


def test_pairwise_symmetry() -> None:
    table_a = FrequencyTable.from_counts("group_a", {"item": 20, "other": 80})
    table_b = FrequencyTable.from_counts("group_b", {"item": 20, "other": 180})
    options = PairwiseComparisonOptions(
        scale=10000,
        zero_handling=ZeroHandling(ZeroHandlingMode.ZERO_ONLY, 0.5),
    )

    original = {row.item: row for row in compare_pair(table_a, table_b, options=options).rows}["item"]
    swapped = {row.item: row for row in compare_pair(table_b, table_a, options=options).rows}["item"]

    assert swapped.rate_a == pytest.approx(original.rate_b)
    assert swapped.rate_b == pytest.approx(original.rate_a)
    assert swapped.rate_difference == pytest.approx(-original.rate_difference)
    assert swapped.log_ratio == pytest.approx(-original.log_ratio)
    assert swapped.log_likelihood == pytest.approx(original.log_likelihood)
    assert original.direction == "group_a"
    assert swapped.direction == "group_a"
    assert swapped.total_count == pytest.approx(original.total_count)


def test_zero_handling_modes_are_distinct() -> None:
    table_a = FrequencyTable.from_counts("group_a", {"item": 0, "other": 100})
    table_b = FrequencyTable.from_counts("group_b", {"item": 10, "other": 90})

    zero_only = compare_pair(
        table_a,
        table_b,
        options=PairwiseComparisonOptions(
            zero_handling=ZeroHandling(ZeroHandlingMode.ZERO_ONLY, 0.5),
        ),
    )
    additive = compare_pair(
        table_a,
        table_b,
        options=PairwiseComparisonOptions(
            zero_handling=ZeroHandling(ZeroHandlingMode.ADDITIVE, 0.5),
        ),
    )

    zero_row = {row.item: row for row in zero_only.rows}["item"]
    additive_row = {row.item: row for row in additive.rows}["item"]
    assert math.isfinite(zero_row.log_ratio)
    assert zero_row.log_ratio != additive_row.log_ratio


def test_min_total_count_filters_on_summed_count() -> None:
    result = compare_pair(
        FrequencyTable.from_counts("group_a", {"rare": 1, "common": 3}),
        FrequencyTable.from_counts("group_b", {"common": 2}),
        options=PairwiseComparisonOptions(min_total_count=2),
    )

    assert {row.item for row in result.rows} == {"common"}


def test_compare_many_rates_and_range() -> None:
    result = compare_many(
        [
            FrequencyTable.from_counts("a", {"item": 10, "other": 90}),
            FrequencyTable.from_counts("b", {"item": 1, "other": 99}),
            FrequencyTable.from_counts("c", {"item": 5, "other": 95}),
        ],
        scale=1,
    )
    row = {row.item: row for row in result.rows}["item"]

    assert row.counts["a"] == 10
    assert row.rates["a"] == pytest.approx(0.1)
    assert row.max_label == "a"
    assert row.min_label == "b"
    assert row.range_relative == pytest.approx(row.max_rate - row.min_rate)


def test_compare_many_rejects_fewer_than_two_tables() -> None:
    with pytest.raises(ComparisonEngineError):
        compare_many([FrequencyTable.from_counts("a", {"item": 1})])
