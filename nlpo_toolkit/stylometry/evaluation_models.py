from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .errors import StylometryError
from .models import FeatureSelection, InputFormat


def _text(value: str, name: str) -> None:
    if not value.strip():
        raise StylometryError(f"{name} must not be empty")


def _values(values: tuple[float, ...]) -> tuple[float, ...]:
    result = tuple(values)
    for value in result:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise StylometryError("profile values must be finite numbers")
    return tuple(float(value) for value in result)


@dataclass(frozen=True)
class AuthorshipAssignment:
    observation_id: str
    author: str
    work_id: str

    def __post_init__(self) -> None:
        _text(self.observation_id, "metadata ID")
        _text(self.author, "author")
        _text(self.work_id, "work ID")


@dataclass(frozen=True)
class AuthorshipMetadata:
    assignments: tuple[AuthorshipAssignment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        ids = tuple(item.observation_id for item in self.assignments)
        if len(ids) != len(set(ids)):
            raise StylometryError("authorship metadata IDs must be unique")
        work_authors: dict[str, str] = {}
        for item in self.assignments:
            previous = work_authors.setdefault(item.work_id, item.author)
            if previous != item.author:
                raise StylometryError(
                    f"work {item.work_id!r} is assigned to multiple authors"
                )


@dataclass(frozen=True)
class LabeledFeatureObservation:
    identifier: str
    author: str
    work_id: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _text(self.identifier, "observation identifier")
        _text(self.author, "author")
        _text(self.work_id, "work ID")
        object.__setattr__(self, "values", _values(self.values))


@dataclass(frozen=True)
class LabeledFeatureDataset:
    feature_names: tuple[str, ...]
    observations: tuple[LabeledFeatureObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "observations", tuple(self.observations))
        if not self.feature_names or any(not name for name in self.feature_names):
            raise StylometryError("labeled dataset requires feature names")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise StylometryError("labeled feature names must be unique")
        ids = tuple(item.identifier for item in self.observations)
        if len(ids) != len(set(ids)):
            raise StylometryError("labeled observation identifiers must be unique")
        work_authors: dict[str, str] = {}
        for item in self.observations:
            if len(item.values) != len(self.feature_names):
                raise StylometryError(
                    "labeled feature vector width does not match schema"
                )
            previous = work_authors.setdefault(item.work_id, item.author)
            if previous != item.author:
                raise StylometryError(
                    f"work {item.work_id!r} is assigned to multiple authors"
                )

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def work_count(self) -> int:
        return len({item.work_id for item in self.observations})

    @property
    def author_count(self) -> int:
        return len({item.author for item in self.observations})

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class WorkProfile:
    work_id: str
    author: str
    observation_ids: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _text(self.work_id, "work ID")
        _text(self.author, "author")
        object.__setattr__(self, "observation_ids", tuple(self.observation_ids))
        object.__setattr__(self, "values", _values(self.values))


@dataclass(frozen=True)
class AuthorProfile:
    author: str
    training_work_ids: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _text(self.author, "author")
        object.__setattr__(self, "training_work_ids", tuple(self.training_work_ids))
        object.__setattr__(self, "values", _values(self.values))


@dataclass(frozen=True)
class LeaveOneWorkOutFold:
    fold_index: int
    test_work: WorkProfile
    training_works: tuple[WorkProfile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "training_works", tuple(self.training_works))


@dataclass(frozen=True)
class LeaveOneWorkOutEvaluationRequest:
    features_path: Path
    input_format: InputFormat
    feature_selection: FeatureSelection
    metadata_path: Path
    metadata_format: InputFormat
    metadata_id_column: str
    author_column: str
    work_column: str

    def __post_init__(self) -> None:
        if self.input_format not in ("csv", "tsv"):
            raise StylometryError("input format must be 'csv' or 'tsv'")
        if self.metadata_format not in ("csv", "tsv"):
            raise StylometryError("metadata format must be 'csv' or 'tsv'")
        _text(self.metadata_id_column, "metadata ID column")
        _text(self.author_column, "author column")
        _text(self.work_column, "work column")
