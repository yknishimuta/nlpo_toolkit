from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .errors import StylometryError
from .models import FeatureSelection, InputFormat


class VerificationDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


class VerificationCalibrationKind(str, Enum):
    GENUINE = "genuine"
    IMPOSTOR = "impostor"


@dataclass(frozen=True)
class VerificationThresholdSettings:
    genuine_quantile: float = 0.95
    impostor_quantile: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (
            ("genuine_quantile", self.genuine_quantile),
            ("impostor_quantile", self.impostor_quantile),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise StylometryError(f"{name} must be finite and between 0 and 1")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class VerificationRequest:
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
    thresholds: VerificationThresholdSettings = VerificationThresholdSettings()

    def __post_init__(self) -> None:
        if not isinstance(self.features_path, Path) or not isinstance(
            self.metadata_path, Path
        ):
            raise StylometryError("verification input paths must be Path values")
        if self.input_format not in ("csv", "tsv"):
            raise StylometryError("input format must be 'csv' or 'tsv'")
        if self.metadata_format not in ("csv", "tsv"):
            raise StylometryError("metadata format must be 'csv' or 'tsv'")
        for value, name in (
            (self.metadata_id_column, "metadata ID column"),
            (self.author_column, "author column"),
            (self.work_column, "work column"),
            (self.candidate_author, "candidate author"),
            (self.query_work, "query work"),
        ):
            if not value.strip():
                raise StylometryError(f"{name} must not be empty")
