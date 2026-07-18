from __future__ import annotations

import math
from collections.abc import Sequence

from .delta import burrows_delta
from .errors import StylometryError
from .evaluation import work_feature_dataset
from .evaluation_models import WorkProfile
from .models import StandardizedObservation
from .standardization import fit_zscore_model, transform_feature_dataset
from .verification_models import (
    VerificationCalibrationKind, VerificationDecision, VerificationThresholdSettings,
)
from .verification_results import (
    VerificationCalibrationScore, VerificationDistributionSummary,
    VerificationNearestBackground, VerificationResult, VerificationThresholds,
)


def linear_quantile(values: Sequence[float], quantile: float) -> float:
    if (
        isinstance(quantile, bool) or not isinstance(quantile, (int, float))
        or not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0
    ):
        raise StylometryError("quantile must be finite and between 0 and 1")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise StylometryError("quantile values must be non-empty and finite")
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise StylometryError("quantile values must be non-empty and finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _centroid(
    observations: Sequence[StandardizedObservation], identifier: str
) -> StandardizedObservation:
    return StandardizedObservation(
        identifier,
        tuple(
            sum(item.values[index] for item in observations) / len(observations)
            for index in range(len(observations[0].values))
        ),
    )


def classify_verification_distance(
    distance: float, *, accept_threshold: float, reject_threshold: float
) -> VerificationDecision:
    if distance < accept_threshold:
        return VerificationDecision.ACCEPT
    if distance > reject_threshold:
        return VerificationDecision.REJECT
    return VerificationDecision.INCONCLUSIVE


def _distribution(
    values: tuple[float, ...], quantile: float
) -> VerificationDistributionSummary:
    return VerificationDistributionSummary(
        len(values), min(values), linear_quantile(values, 0.5), max(values),
        quantile, linear_quantile(values, quantile),
    )


def evaluate_verification(
    feature_names: tuple[str, ...],
    profiles: tuple[WorkProfile, ...],
    *, candidate_author: str,
    query_work: str,
    settings: VerificationThresholdSettings,
) -> VerificationResult:
    query_profiles = tuple(item for item in profiles if item.work_id == query_work)
    if len(query_profiles) != 1:
        raise StylometryError(f"query work not found: {query_work!r}")
    query = query_profiles[0]
    references = tuple(item for item in profiles if item.work_id != query_work)
    candidates = tuple(item for item in references if item.author == candidate_author)
    background = tuple(item for item in references if item.author != candidate_author)
    if len(candidates) < 3:
        raise StylometryError(
            "verification requires at least three candidate reference works; "
            f"author {candidate_author!r} has {len(candidates)}"
        )
    if len(background) < 2:
        raise StylometryError(
            f"verification requires at least two background works; found {len(background)}"
        )
    reference_dataset = work_feature_dataset(feature_names, references)
    try:
        model = fit_zscore_model(reference_dataset)
    except StylometryError as exc:
        if "all selected features have zero variance" in str(exc):
            raise StylometryError(
                "all selected features have zero variance in verification reference works"
            ) from exc
        raise
    standardized = transform_feature_dataset(reference_dataset, model=model)
    by_id = {item.identifier: item for item in standardized.observations}
    candidate_observations = tuple(by_id[item.work_id] for item in candidates)
    candidate_centroid = _centroid(candidate_observations, candidate_author)
    genuine = tuple(
        VerificationCalibrationScore(
            VerificationCalibrationKind.GENUINE, work.work_id, work.author,
            burrows_delta(
                by_id[work.work_id],
                _centroid(
                    tuple(item for item in candidate_observations if item.identifier != work.work_id),
                    candidate_author,
                ),
            ),
            tuple(sorted(item.work_id for item in candidates if item.work_id != work.work_id)),
        )
        for work in candidates
    )
    impostor = tuple(
        VerificationCalibrationScore(
            VerificationCalibrationKind.IMPOSTOR, work.work_id, work.author,
            burrows_delta(by_id[work.work_id], candidate_centroid),
            tuple(sorted(item.work_id for item in candidates)),
        )
        for work in background
    )
    genuine = tuple(sorted(genuine, key=lambda item: (item.distance, item.author, item.work_id)))
    impostor = tuple(sorted(impostor, key=lambda item: (item.distance, item.author, item.work_id)))
    genuine_values = tuple(item.distance for item in genuine)
    impostor_values = tuple(item.distance for item in impostor)
    genuine_boundary = linear_quantile(genuine_values, settings.genuine_quantile)
    impostor_boundary = linear_quantile(impostor_values, settings.impostor_quantile)
    thresholds = VerificationThresholds(
        settings.genuine_quantile, settings.impostor_quantile,
        genuine_boundary, impostor_boundary,
        min(genuine_boundary, impostor_boundary),
        max(genuine_boundary, impostor_boundary),
    )
    query_observation = transform_feature_dataset(
        work_feature_dataset(feature_names, (query,)), model=model
    ).observations[0]
    query_distance = burrows_delta(query_observation, candidate_centroid)
    nearest_items = tuple(
        sorted(
            ((burrows_delta(query_observation, by_id[item.work_id]), item.author, item.work_id)
             for item in background),
            key=lambda item: (item[0], item[1], item[2]),
        )
    )
    nearest_distance, nearest_author, nearest_work = nearest_items[0]
    return VerificationResult(
        classify_verification_distance(
            query_distance, accept_threshold=thresholds.accept_threshold,
            reject_threshold=thresholds.reject_threshold,
        ),
        candidate_author, query_work, len(query.observation_ids), query_distance,
        tuple(sorted(item.work_id for item in candidates)), len(background),
        len({item.author for item in background}), model.input_feature_names,
        model.retained_feature_names, model.dropped_zero_variance_features,
        thresholds, _distribution(genuine_values, settings.genuine_quantile),
        _distribution(impostor_values, settings.impostor_quantile),
        VerificationNearestBackground(
            nearest_work, nearest_author, nearest_distance,
            nearest_distance - query_distance,
        ),
        genuine + impostor,
    )
