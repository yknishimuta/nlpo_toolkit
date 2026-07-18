from __future__ import annotations

import math

import pytest

from nlpo_toolkit.stylometry.errors import StylometryError, StylometryMetricError
from nlpo_toolkit.stylometry.metrics import (
    ScorePreference,
    StylometryMetric,
    burrows_delta,
    compute_stylometry_score,
    cosine_delta,
    cosine_similarity,
    manhattan_distance,
    score_preference,
)
from nlpo_toolkit.stylometry.models import (
    StandardizedFeatureDataset,
    StandardizedObservation,
)
from nlpo_toolkit.stylometry.neighbor_models import NeighborRankingRequest
from nlpo_toolkit.stylometry.models import FeatureSelection
from nlpo_toolkit.stylometry.ranking import build_neighbor_rankings


def test_metric_values_and_preferences() -> None:
    assert tuple(metric.value for metric in StylometryMetric) == (
        "burrows_delta",
        "manhattan",
        "cosine_delta",
        "cosine_similarity",
    )
    assert (
        score_preference(StylometryMetric.BURROWS_DELTA)
        is ScorePreference.LOWER_IS_BETTER
    )
    assert (
        score_preference(StylometryMetric.MANHATTAN) is ScorePreference.LOWER_IS_BETTER
    )
    assert (
        score_preference(StylometryMetric.COSINE_DELTA)
        is ScorePreference.LOWER_IS_BETTER
    )
    assert (
        score_preference(StylometryMetric.COSINE_SIMILARITY)
        is ScorePreference.HIGHER_IS_BETTER
    )
    with pytest.raises(StylometryMetricError):
        score_preference("bad")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "left,right",
    [([], []), ([1], [1, 2]), ([math.nan], [1]), ([math.inf], [1]), ([True], [1])],
)
def test_common_vector_validation(left, right) -> None:
    with pytest.raises(StylometryMetricError):
        manhattan_distance(left, right)


def test_burrows_and_manhattan_definitions_without_mutation() -> None:
    left = [1.0, 2.0]
    right = [4.0, 6.0]
    assert manhattan_distance(left, right) == 7.0
    assert burrows_delta(left, right) == 3.5
    assert manhattan_distance(left, right) == burrows_delta(left, right) * 2
    assert manhattan_distance(left, left) == burrows_delta(left, left) == 0.0
    assert manhattan_distance(left, right) == manhattan_distance(right, left)
    assert left == [1.0, 2.0]
    assert right == [4.0, 6.0]


def test_cosine_similarity_and_delta_definitions() -> None:
    assert cosine_similarity((1, 0), (1, 0)) == pytest.approx(1.0)
    assert cosine_similarity((1, 0), (0, 1)) == pytest.approx(0.0)
    assert cosine_similarity((1, 0), (-1, 0)) == pytest.approx(-1.0)
    assert cosine_similarity((1, 2), (2, 4)) == pytest.approx(1.0)
    assert cosine_similarity((1, 2), (2, 4)) == cosine_similarity((2, 4), (1, 2))
    assert cosine_delta((1, 0), (1, 0)) == pytest.approx(0.0)
    assert cosine_delta((1, 0), (0, 1)) == pytest.approx(1.0)
    assert cosine_delta((1, 0), (-1, 0)) == pytest.approx(2.0)
    assert cosine_delta((1, 2), (2, 4)) == pytest.approx(
        1 - cosine_similarity((1, 2), (2, 4))
    )
    for left, right in (((0, 0), (1, 0)), ((1, 0), (0, 0)), ((0, 0), (0, 0))):
        with pytest.raises(StylometryMetricError, match="zero-norm"):
            cosine_similarity(left, right)


def test_dispatches_all_metrics_and_rejects_unknown() -> None:
    left, right = (1.0, 0.0), (0.0, 1.0)
    assert (
        compute_stylometry_score(left, right, metric=StylometryMetric.BURROWS_DELTA)
        == 1.0
    )
    assert (
        compute_stylometry_score(left, right, metric=StylometryMetric.MANHATTAN) == 2.0
    )
    assert (
        compute_stylometry_score(left, right, metric=StylometryMetric.COSINE_DELTA)
        == 1.0
    )
    assert (
        compute_stylometry_score(left, right, metric=StylometryMetric.COSINE_SIMILARITY)
        == 0.0
    )
    with pytest.raises(StylometryMetricError):
        compute_stylometry_score(left, right, metric="bad")  # type: ignore[arg-type]


def _dataset() -> StandardizedFeatureDataset:
    return StandardizedFeatureDataset(
        ("f1", "f2"),
        (
            StandardizedObservation("query", (1.0, 0.0)),
            StandardizedObservation("B", (1.0, 1.0)),
            StandardizedObservation("A", (1.0, -1.0)),
        ),
    )


def test_ranking_direction_ties_top_and_directed_rows() -> None:
    distance = build_neighbor_rankings(_dataset(), metric=StylometryMetric.MANHATTAN)
    assert [item.query_id for item in distance] == ["query", "B", "A"]
    assert [item.neighbor_id for item in distance[0].neighbors] == ["A", "B"]
    assert sum(len(item.neighbors) for item in distance) == 6
    similarity = build_neighbor_rankings(
        _dataset(), metric=StylometryMetric.COSINE_SIMILARITY, top=1
    )
    assert similarity[0].neighbors[0].neighbor_id == "A"
    assert all(len(item.neighbors) == 1 for item in similarity)
    assert (
        build_neighbor_rankings(_dataset(), metric=StylometryMetric.MANHATTAN, top=99)
        == distance
    )


def test_ranking_validation_and_cosine_zero_id() -> None:
    one = StandardizedFeatureDataset(("f",), (StandardizedObservation("only", (1.0,)),))
    with pytest.raises(StylometryMetricError, match="at least two"):
        build_neighbor_rankings(one, metric=StylometryMetric.MANHATTAN)
    for top in (0, -1, True):
        with pytest.raises(StylometryMetricError, match="positive"):
            build_neighbor_rankings(
                _dataset(), metric=StylometryMetric.MANHATTAN, top=top
            )
    zero = StandardizedFeatureDataset(
        ("f",),
        (
            StandardizedObservation("zero", (0.0,)),
            StandardizedObservation("other", (1.0,)),
        ),
    )
    with pytest.raises(StylometryMetricError, match="'zero'"):
        build_neighbor_rankings(zero, metric=StylometryMetric.COSINE_DELTA)
    assert build_neighbor_rankings(zero, metric=StylometryMetric.MANHATTAN)


def test_neighbor_request_validates_top_and_metric(tmp_path) -> None:
    selection = FeatureSelection(prefixes=("mfw_",))
    for top in (0, -1, True):
        with pytest.raises(StylometryError, match="positive"):
            NeighborRankingRequest(tmp_path / "x", "csv", selection, top=top)
    with pytest.raises(StylometryError, match="StylometryMetric"):
        NeighborRankingRequest(
            tmp_path / "x",
            "csv",
            selection,
            metric="bad",  # type: ignore[arg-type]
        )
