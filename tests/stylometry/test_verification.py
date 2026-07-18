from __future__ import annotations

import math

import pytest

from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.evaluation_models import WorkProfile
from nlpo_toolkit.stylometry.verification import (
    classify_verification_distance,
    evaluate_verification,
    linear_quantile,
)
from nlpo_toolkit.stylometry.verification_models import (
    VerificationDecision,
    VerificationThresholdSettings,
)


def _profiles(query: float = 1.0, query_author: str = "unknown") -> tuple[WorkProfile, ...]:
    values = (
        ("a1", "candidate", 0.0),
        ("a2", "candidate", 1.0),
        ("a3", "candidate", 2.0),
        ("b1", "background_b", 8.0),
        ("b2", "background_c", 10.0),
        ("query", query_author, query),
    )
    return tuple(WorkProfile(work, author, (f"{work}_sample",), (value,)) for work, author, value in values)


@pytest.mark.parametrize(
    ("values", "quantile", "expected"),
    [
        ((4.0,), 0.5, 4.0),
        ((1.0, 2.0, 3.0), 0.5, 2.0),
        ((1.0, 2.0, 3.0, 4.0), 0.5, 2.5),
        ((1.0, 2.0, 3.0), 0.0, 1.0),
        ((1.0, 2.0, 3.0), 1.0, 3.0),
        ((0.0, 10.0), 0.05, 0.5),
        ((0.0, 10.0), 0.95, 9.5),
    ],
)
def test_linear_quantile(values, quantile, expected) -> None:
    assert linear_quantile(values, quantile) == pytest.approx(expected)


@pytest.mark.parametrize("quantile", (-0.1, 1.1, math.nan, math.inf, True))
def test_linear_quantile_rejects_invalid_quantile(quantile) -> None:
    with pytest.raises(StylometryError):
        linear_quantile((1.0,), quantile)


@pytest.mark.parametrize("values", ((), (math.nan,), (math.inf,), (True,)))
def test_linear_quantile_rejects_invalid_values(values) -> None:
    with pytest.raises(StylometryError):
        linear_quantile(values, 0.5)


def test_classification_is_conservative_at_boundaries_and_handles_overlap() -> None:
    assert classify_verification_distance(
        0.9, accept_threshold=1.0, reject_threshold=2.0
    ) is VerificationDecision.ACCEPT
    assert classify_verification_distance(
        2.1, accept_threshold=1.0, reject_threshold=2.0
    ) is VerificationDecision.REJECT
    assert classify_verification_distance(
        1.5, accept_threshold=1.0, reject_threshold=2.0
    ) is VerificationDecision.INCONCLUSIVE
    for boundary in (1.0, 2.0):
        assert classify_verification_distance(
            boundary, accept_threshold=1.0, reject_threshold=2.0
        ) is VerificationDecision.INCONCLUSIVE


def test_query_is_excluded_from_fit_and_calibration(monkeypatch) -> None:
    from nlpo_toolkit.stylometry import verification

    original = verification.fit_zscore_model
    seen: list[tuple[str, ...]] = []

    def recording_fit(dataset):
        seen.append(tuple(item.identifier for item in dataset.observations))
        return original(dataset)

    monkeypatch.setattr(verification, "fit_zscore_model", recording_fit)
    first = evaluate_verification(
        ("f1",), _profiles(1.0), candidate_author="candidate",
        query_work="query", settings=VerificationThresholdSettings(),
    )
    second = evaluate_verification(
        ("f1",), _profiles(1000.0), candidate_author="candidate",
        query_work="query", settings=VerificationThresholdSettings(),
    )
    assert seen == [("a1", "a2", "a3", "b1", "b2")] * 2
    assert first.thresholds == second.thresholds
    assert first.calibration_scores == second.calibration_scores
    assert first.query_distance != second.query_distance


def test_calibration_centroids_exclude_self_and_impostors_use_full_candidate() -> None:
    result = evaluate_verification(
        ("f1",), _profiles(), candidate_author="candidate",
        query_work="query", settings=VerificationThresholdSettings(),
    )
    genuine = result.calibration_scores[:3]
    impostor = result.calibration_scores[3:]
    assert all(item.work_id not in item.centroid_work_ids for item in genuine)
    assert {item.centroid_work_ids for item in impostor} == {("a1", "a2", "a3")}
    assert result.thresholds.accept_threshold <= result.thresholds.reject_threshold


def test_query_metadata_author_is_ignored() -> None:
    unknown = evaluate_verification(
        ("f1",), _profiles(query_author="unknown"), candidate_author="candidate",
        query_work="query", settings=VerificationThresholdSettings(),
    )
    mislabeled = evaluate_verification(
        ("f1",), _profiles(query_author="candidate"), candidate_author="candidate",
        query_work="query", settings=VerificationThresholdSettings(),
    )
    assert unknown == mislabeled


def test_minimum_reference_requirements_and_zero_variance() -> None:
    with pytest.raises(StylometryError, match="three candidate"):
        evaluate_verification(
            ("f",), _profiles()[:2] + _profiles()[3:], candidate_author="candidate",
            query_work="query", settings=VerificationThresholdSettings(),
        )
    constant = tuple(
        WorkProfile(item.work_id, item.author, item.observation_ids, (1.0,))
        for item in _profiles()
    )
    with pytest.raises(StylometryError, match="zero variance in verification"):
        evaluate_verification(
            ("f",), constant, candidate_author="candidate", query_work="query",
            settings=VerificationThresholdSettings(),
        )
