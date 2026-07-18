from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import StylometryError
from .verification_models import VerificationCalibrationKind, VerificationDecision


def _finite(value: float, name: str, *, nonnegative: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StylometryError(f"{name} must be finite and non-negative")
    if not math.isfinite(value) or (nonnegative and value < 0.0):
        raise StylometryError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class VerificationCalibrationScore:
    kind: VerificationCalibrationKind
    work_id: str
    author: str
    distance: float
    centroid_work_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "centroid_work_ids", tuple(self.centroid_work_ids))
        if (
            not self.work_id
            or not self.author
            or not self.centroid_work_ids
            or any(not item for item in self.centroid_work_ids)
        ):
            raise StylometryError("verification calibration labels must not be empty")
        if len(self.centroid_work_ids) != len(set(self.centroid_work_ids)):
            raise StylometryError("verification centroid works must be unique")
        _finite(self.distance, "calibration distance")


@dataclass(frozen=True)
class VerificationDistributionSummary:
    count: int
    minimum: float
    median: float
    maximum: float
    selected_quantile: float
    selected_quantile_value: float

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise StylometryError("verification distribution must not be empty")
        for name in ("minimum", "median", "maximum", "selected_quantile_value"):
            _finite(getattr(self, name), name)
        if not 0.0 <= self.selected_quantile <= 1.0:
            raise StylometryError("selected quantile must be between 0 and 1")


@dataclass(frozen=True)
class VerificationThresholds:
    genuine_quantile: float
    impostor_quantile: float
    genuine_boundary: float
    impostor_boundary: float
    accept_threshold: float
    reject_threshold: float

    def __post_init__(self) -> None:
        for name in (
            "genuine_boundary", "impostor_boundary", "accept_threshold",
            "reject_threshold",
        ):
            _finite(getattr(self, name), name)
        if self.accept_threshold > self.reject_threshold:
            raise StylometryError("verification thresholds are out of order")


@dataclass(frozen=True)
class VerificationNearestBackground:
    work_id: str
    author: str
    distance: float
    candidate_vs_background_margin: float

    def __post_init__(self) -> None:
        if not self.work_id or not self.author:
            raise StylometryError("nearest background labels must not be empty")
        _finite(self.distance, "nearest background distance")
        _finite(
            self.candidate_vs_background_margin,
            "candidate/background margin",
            nonnegative=False,
        )


@dataclass(frozen=True)
class VerificationResult:
    decision: VerificationDecision
    candidate_author: str
    query_work: str
    query_sample_count: int
    query_distance: float
    candidate_reference_works: tuple[str, ...]
    background_work_count: int
    background_author_count: int
    input_feature_names: tuple[str, ...]
    retained_feature_names: tuple[str, ...]
    dropped_zero_variance_features: tuple[str, ...]
    thresholds: VerificationThresholds
    genuine_distribution: VerificationDistributionSummary
    impostor_distribution: VerificationDistributionSummary
    nearest_background: VerificationNearestBackground
    calibration_scores: tuple[VerificationCalibrationScore, ...]

    def __post_init__(self) -> None:
        for name in (
            "candidate_reference_works", "input_feature_names",
            "retained_feature_names", "dropped_zero_variance_features",
            "calibration_scores",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.candidate_author or not self.query_work:
            raise StylometryError("verification labels must not be empty")
        if self.query_sample_count <= 0:
            raise StylometryError("query sample count must be positive")
        _finite(self.query_distance, "query distance")

    @property
    def candidate_reference_work_count(self) -> int:
        return len(self.candidate_reference_works)

    @property
    def reference_work_count(self) -> int:
        return self.candidate_reference_work_count + self.background_work_count
