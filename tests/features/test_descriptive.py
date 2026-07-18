from __future__ import annotations

import pytest

from nlpo_toolkit.corpus_analysis.features.descriptive import (
    DistributionSummary,
    summarize_distribution,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        ((), DistributionSummary(0.0, 0.0, 0.0, 0.0)),
        ((4,), DistributionSummary(0.0, 4.0, 4.0, 4.0)),
        ((1, 2, 3), DistributionSummary(2 / 3, 2.0, 1.5, 2.5)),
        ((1, 2, 3, 4), DistributionSummary(1.25, 2.5, 1.75, 3.25)),
        ((5, 5, 5), DistributionSummary(0.0, 5.0, 5.0, 5.0)),
    ),
)
def test_distribution_summary_uses_population_variance_and_linear_quantiles(
    values: tuple[int, ...],
    expected: DistributionSummary,
) -> None:
    actual = summarize_distribution(values)

    assert actual.variance == pytest.approx(expected.variance)
    assert actual.median == pytest.approx(expected.median)
    assert actual.q25 == pytest.approx(expected.q25)
    assert actual.q75 == pytest.approx(expected.q75)
    assert all(
        isinstance(value, float)
        for value in (actual.variance, actual.median, actual.q25, actual.q75)
    )


def test_distribution_summary_is_order_independent_and_does_not_mutate_input() -> None:
    values = [4, 1, 3, 2]

    actual = summarize_distribution(values)

    assert values == [4, 1, 3, 2]
    assert actual == summarize_distribution(tuple(reversed(values)))
