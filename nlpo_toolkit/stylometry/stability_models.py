from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .errors import StylometryError
from .models import FeatureSelection, InputFormat
from .verification_models import VerificationThresholdSettings


class ResamplingAxis(str, Enum):
    WORKS = "works"
    SAMPLES = "samples"
    FEATURES = "features"


class VerificationStabilityStatus(str, Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"


class ResamplingAttemptStatus(str, Enum):
    SUCCESS = "success"
    REJECTED = "rejected"


class ReferenceWorkRole(str, Enum):
    CANDIDATE = "candidate"
    BACKGROUND = "background"


def _fraction(value: float, name: str, *, allow_zero: bool = False) -> float:
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not lower_ok
        or value > 1.0
    ):
        boundary = "0 <= value <= 1" if allow_zero else "0 < value <= 1"
        raise StylometryError(f"{name} must satisfy {boundary}")
    return float(value)


@dataclass(frozen=True)
class ResamplingIntervalSettings:
    lower: float = 0.025
    upper: float = 0.975

    def __post_init__(self) -> None:
        lower = _fraction(self.lower, "interval lower", allow_zero=True)
        upper = _fraction(self.upper, "interval upper", allow_zero=True)
        if not lower < 0.5 < upper:
            raise StylometryError("interval bounds must satisfy lower < 0.5 < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class VerificationStabilitySettings:
    axes: tuple[ResamplingAxis, ...]
    iterations: int = 1000
    seed: int = 0
    max_attempts: int | None = None
    work_fraction: float = 0.8
    feature_fraction: float = 0.8
    stability_threshold: float = 0.8
    interval: ResamplingIntervalSettings = ResamplingIntervalSettings()

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        if not self.axes or len(self.axes) != len(set(self.axes)):
            raise StylometryError("resampling axes must be non-empty and unique")
        if any(not isinstance(axis, ResamplingAxis) for axis in self.axes):
            raise StylometryError("invalid resampling axis")
        for name, value in (("iterations", self.iterations), ("seed", self.seed)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise StylometryError(f"{name} must be an integer")
        if not 1 <= self.iterations <= 1_000_000:
            raise StylometryError("iterations must be between 1 and 1000000")
        maximum = (
            self.iterations * 10 if self.max_attempts is None else self.max_attempts
        )
        if isinstance(maximum, bool) or not isinstance(maximum, int):
            raise StylometryError("max attempts must be an integer")
        if not self.iterations <= maximum <= 10_000_000:
            raise StylometryError(
                "max attempts must be at least iterations and at most 10000000"
            )
        object.__setattr__(self, "max_attempts", maximum)
        object.__setattr__(
            self, "work_fraction", _fraction(self.work_fraction, "work fraction")
        )
        object.__setattr__(
            self,
            "feature_fraction",
            _fraction(self.feature_fraction, "feature fraction"),
        )
        object.__setattr__(
            self,
            "stability_threshold",
            _fraction(self.stability_threshold, "stability threshold"),
        )


@dataclass(frozen=True)
class VerificationStabilityRequest:
    features_path: Path
    metadata_path: Path
    input_format: InputFormat
    metadata_format: InputFormat
    feature_selection: FeatureSelection
    metadata_id_column: str
    author_column: str
    work_column: str
    candidate_author: str
    query_work: str
    verification_thresholds: VerificationThresholdSettings
    stability: VerificationStabilitySettings

    def __post_init__(self) -> None:
        if not isinstance(self.features_path, Path) or not isinstance(
            self.metadata_path, Path
        ):
            raise StylometryError("stability input paths must be Path values")
        if self.input_format not in ("csv", "tsv") or self.metadata_format not in (
            "csv",
            "tsv",
        ):
            raise StylometryError("stability input formats must be 'csv' or 'tsv'")
        values = (
            self.metadata_id_column,
            self.author_column,
            self.work_column,
            self.candidate_author,
            self.query_work,
        )
        if any(not value.strip() for value in values):
            raise StylometryError("stability column and label values must not be empty")
