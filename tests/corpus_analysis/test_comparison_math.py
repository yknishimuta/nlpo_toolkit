from __future__ import annotations

import math
from collections import Counter

import pytest

from nlpo_toolkit.corpus_analysis.comparison import (
    ComparisonSpec,
    calculate_log_likelihood,
    calculate_log_ratio,
    compare_counters,
)


def test_log_ratio_same_relative_frequency_is_zero() -> None:
    assert calculate_log_ratio(
        count_a=10,
        tokens_a=100,
        count_b=20,
        tokens_b=200,
        zero_correction=0.5,
    ) == pytest.approx(0.0)


def test_log_ratio_group_a_twice_is_one() -> None:
    assert calculate_log_ratio(
        count_a=20,
        tokens_a=100,
        count_b=20,
        tokens_b=200,
        zero_correction=0.5,
    ) == pytest.approx(1.0)


def test_log_ratio_group_b_twice_is_minus_one() -> None:
    assert calculate_log_ratio(
        count_a=10,
        tokens_a=100,
        count_b=40,
        tokens_b=200,
        zero_correction=0.5,
    ) == pytest.approx(-1.0)


def test_log_ratio_zero_counts_are_finite_and_correction_changes_result() -> None:
    zero_a = calculate_log_ratio(
        count_a=0,
        tokens_a=100,
        count_b=10,
        tokens_b=100,
        zero_correction=0.5,
    )
    zero_b = calculate_log_ratio(
        count_a=10,
        tokens_a=100,
        count_b=0,
        tokens_b=100,
        zero_correction=0.5,
    )
    changed = calculate_log_ratio(
        count_a=0,
        tokens_a=100,
        count_b=10,
        tokens_b=100,
        zero_correction=1.0,
    )

    assert math.isfinite(zero_a)
    assert math.isfinite(zero_b)
    assert changed != zero_a


def test_log_ratio_swap_reverses_sign() -> None:
    original = calculate_log_ratio(
        count_a=5,
        tokens_a=100,
        count_b=20,
        tokens_b=100,
        zero_correction=0.5,
    )
    swapped = calculate_log_ratio(
        count_a=20,
        tokens_a=100,
        count_b=5,
        tokens_b=100,
        zero_correction=0.5,
    )

    assert swapped == pytest.approx(-original)


def test_log_likelihood_same_distribution_is_zero() -> None:
    assert calculate_log_likelihood(
        count_a=10,
        tokens_a=100,
        count_b=20,
        tokens_b=200,
    ) == pytest.approx(0.0)


def test_log_likelihood_difference_is_positive_and_symmetric() -> None:
    original = calculate_log_likelihood(
        count_a=20,
        tokens_a=100,
        count_b=20,
        tokens_b=200,
    )
    swapped = calculate_log_likelihood(
        count_a=20,
        tokens_a=200,
        count_b=20,
        tokens_b=100,
    )

    assert original > 0
    assert swapped == pytest.approx(original)


def test_log_likelihood_allows_observed_zero() -> None:
    value = calculate_log_likelihood(
        count_a=0,
        tokens_a=100,
        count_b=10,
        tokens_b=100,
    )

    assert value > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count_a": 101, "tokens_a": 100, "count_b": 1, "tokens_b": 100},
        {"count_a": -1, "tokens_a": 100, "count_b": 1, "tokens_b": 100},
        {"count_a": 1, "tokens_a": 0, "count_b": 1, "tokens_b": 100},
    ],
)
def test_log_likelihood_rejects_invalid_counts_and_tokens(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        calculate_log_likelihood(**kwargs)


def _test_counters() -> tuple[Counter[str], Counter[str]]:
    return (
        Counter({"item_common": 10, "item_a": 8, "item_rare_a": 1}),
        Counter({"item_common": 20, "item_b": 7, "item_rare_b": 1}),
    )


def test_compare_counters_uses_union_and_keeps_one_sided_items() -> None:
    counter_a, counter_b = _test_counters()
    result = compare_counters(
        counter_a=counter_a,
        counter_b=counter_b,
        spec=ComparisonSpec("comparison_1", "group_a", "group_b"),
        analysis_unit="lemma",
    )

    items = {row.item for row in result.rows}
    assert items == {"item_common", "item_a", "item_rare_a", "item_b", "item_rare_b"}
    assert result.vocabulary_union_size == 5
    assert {row.item for row in result.rows if row.group_b_count == 0} == {
        "item_a",
        "item_rare_a",
    }


def test_compare_counters_total_count_filter_rates_and_direction() -> None:
    counter_a, counter_b = _test_counters()
    result = compare_counters(
        counter_a=counter_a,
        counter_b=counter_b,
        spec=ComparisonSpec(
            "comparison_1",
            "group_a",
            "group_b",
            min_total_count=2,
        ),
        analysis_unit="surface",
    )

    rows = {row.item: row for row in result.rows}
    assert "item_rare_a" not in rows
    assert "item_rare_b" not in rows
    assert rows["item_common"].total_count == 30
    assert rows["item_a"].group_a_rate == pytest.approx(8 / 19 * 10000)
    assert rows["item_a"].direction == "group_a"
    assert rows["item_b"].direction == "group_b"
    assert result.analysis_unit == "surface"


def test_compare_counters_equal_direction() -> None:
    result = compare_counters(
        counter_a=Counter({"item_common": 10}),
        counter_b=Counter({"item_common": 20}),
        spec=ComparisonSpec("comparison_1", "group_a", "group_b"),
        analysis_unit="lemma",
    )

    assert result.rows[0].direction == "equal"


def test_compare_counters_default_sort_is_specified_order() -> None:
    result = compare_counters(
        counter_a=Counter({"item_a": 4, "item_b": 2, "item_c": 1}),
        counter_b=Counter({"item_a": 1, "item_b": 2, "item_c": 4}),
        spec=ComparisonSpec("comparison_1", "group_a", "group_b"),
        analysis_unit="lemma",
    )

    expected = sorted(
        result.rows,
        key=lambda row: (-row.log_likelihood, -abs(row.log_ratio), -row.total_count, row.item),
    )
    assert list(result.rows) == expected


def test_compare_counters_lemma_and_surface_modes() -> None:
    for unit in ("lemma", "surface"):
        result = compare_counters(
            counter_a=Counter({"item_a": 1}),
            counter_b=Counter({"item_b": 1}),
            spec=ComparisonSpec("comparison_1", "group_a", "group_b"),
            analysis_unit=unit,
        )
        assert result.analysis_unit == unit
