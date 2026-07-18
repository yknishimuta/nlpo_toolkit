from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .errors import StylometryError


InputFormat = Literal["csv", "tsv"]


def _validate_names(names: tuple[str, ...], *, kind: str) -> None:
    if any(not name.strip() for name in names):
        raise StylometryError(f"{kind} must not be empty")
    if len(names) != len(set(names)):
        raise StylometryError(f"duplicate {kind}")


def _validate_values(values: tuple[float, ...]) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StylometryError("feature values must be numbers, not bools")
        if not math.isfinite(value):
            raise StylometryError("feature values must be finite")


@dataclass(frozen=True)
class FeatureSelection:
    id_column: str = "group"
    prefixes: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefixes", tuple(self.prefixes))
        object.__setattr__(self, "columns", tuple(self.columns))
        if not self.id_column.strip():
            raise StylometryError("identifier column must not be empty")
        _validate_names(self.prefixes, kind="feature prefix")
        _validate_names(self.columns, kind="feature column")
        if not self.prefixes and not self.columns:
            raise StylometryError(
                "at least one --feature-prefix or --feature-column is required"
            )


@dataclass(frozen=True)
class FeatureObservation:
    identifier: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not self.identifier.strip():
            raise StylometryError("observation identifier must not be empty")
        _validate_values(self.values)
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))


@dataclass(frozen=True)
class FeatureDataset:
    feature_names: tuple[str, ...]
    observations: tuple[FeatureObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "observations", tuple(self.observations))
        if not self.feature_names:
            raise StylometryError("feature dataset must contain at least one feature")
        _validate_names(self.feature_names, kind="feature name")
        identifiers = tuple(item.identifier for item in self.observations)
        if len(identifiers) != len(set(identifiers)):
            raise StylometryError("observation identifiers must be unique")
        for item in self.observations:
            if len(item.values) != len(self.feature_names):
                raise StylometryError("feature vector width does not match schema")

    @property
    def sample_count(self) -> int:
        return len(self.observations)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class ZScoreModel:
    input_feature_names: tuple[str, ...]
    retained_feature_names: tuple[str, ...]
    retained_indices: tuple[int, ...]
    dropped_zero_variance_features: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in (
            "input_feature_names",
            "retained_feature_names",
            "retained_indices",
            "dropped_zero_variance_features",
            "means",
            "standard_deviations",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not (
            len(self.retained_feature_names)
            == len(self.retained_indices)
            == len(self.means)
            == len(self.standard_deviations)
        ):
            raise StylometryError("z-score model widths do not match")

    @property
    def retained_feature_count(self) -> int:
        return len(self.retained_feature_names)


@dataclass(frozen=True)
class StandardizedObservation:
    identifier: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not self.identifier.strip():
            raise StylometryError("observation identifier must not be empty")
        _validate_values(self.values)
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))


@dataclass(frozen=True)
class StandardizedFeatureDataset:
    feature_names: tuple[str, ...]
    observations: tuple[StandardizedObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "observations", tuple(self.observations))
        if not self.feature_names:
            raise StylometryError(
                "standardized dataset must contain at least one feature"
            )
        _validate_names(self.feature_names, kind="feature name")
        for item in self.observations:
            if len(item.values) != len(self.feature_names):
                raise StylometryError("standardized vector width does not match schema")


@dataclass(frozen=True)
class BurrowsDeltaRequest:
    features_path: Path
    input_format: InputFormat
    selection: FeatureSelection

    def __post_init__(self) -> None:
        if not isinstance(self.features_path, Path):
            raise StylometryError("features_path must be a Path")
        if self.input_format not in {"csv", "tsv"}:
            raise StylometryError("input_format must be 'csv' or 'tsv'")
