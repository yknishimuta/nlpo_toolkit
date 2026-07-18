from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import StylometryError


@dataclass(frozen=True)
class AuthorCandidateDistance:
    author: str
    distance: float

    def __post_init__(self) -> None:
        if not self.author or not math.isfinite(self.distance) or self.distance < 0:
            raise StylometryError(
                "candidate author distance must be finite and non-negative"
            )


@dataclass(frozen=True)
class LeaveOneWorkOutFoldResult:
    fold_index: int
    work_id: str
    actual_author: str
    test_observation_ids: tuple[str, ...]
    training_work_ids: tuple[str, ...]
    retained_feature_names: tuple[str, ...]
    dropped_zero_variance_features: tuple[str, ...]
    candidates: tuple[AuthorCandidateDistance, ...]

    def __post_init__(self) -> None:
        for name in (
            "test_observation_ids",
            "training_work_ids",
            "retained_feature_names",
            "dropped_zero_variance_features",
            "candidates",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.candidates) < 2:
            raise StylometryError("LOWO fold requires at least two candidate authors")

    @property
    def predicted_author(self) -> str:
        return self.candidates[0].author

    @property
    def is_correct(self) -> bool:
        return self.predicted_author == self.actual_author

    @property
    def test_sample_count(self) -> int:
        return len(self.test_observation_ids)

    @property
    def training_work_count(self) -> int:
        return len(self.training_work_ids)

    @property
    def candidate_author_count(self) -> int:
        return len(self.candidates)

    @property
    def retained_feature_count(self) -> int:
        return len(self.retained_feature_names)

    @property
    def dropped_feature_count(self) -> int:
        return len(self.dropped_zero_variance_features)

    @property
    def best_distance(self) -> float:
        return self.candidates[0].distance

    @property
    def runner_up_author(self) -> str:
        return self.candidates[1].author

    @property
    def runner_up_distance(self) -> float:
        return self.candidates[1].distance

    @property
    def margin(self) -> float:
        return self.runner_up_distance - self.best_distance


@dataclass(frozen=True)
class AuthorEvaluationSummary:
    author: str
    work_count: int
    correct_work_count: int

    @property
    def accuracy(self) -> float:
        return self.correct_work_count / self.work_count


@dataclass(frozen=True)
class LeaveOneWorkOutSummary:
    per_author: tuple[AuthorEvaluationSummary, ...]
    folds: tuple[LeaveOneWorkOutFoldResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_author", tuple(self.per_author))
        object.__setattr__(self, "folds", tuple(self.folds))

    @property
    def work_count(self) -> int:
        return len(self.folds)

    @property
    def correct_work_count(self) -> int:
        return sum(fold.is_correct for fold in self.folds)

    @property
    def accuracy(self) -> float:
        return self.correct_work_count / self.work_count

    @property
    def author_count(self) -> int:
        return len(self.per_author)

    @property
    def macro_author_accuracy(self) -> float:
        return sum(item.accuracy for item in self.per_author) / self.author_count


@dataclass(frozen=True)
class LeaveOneWorkOutEvaluationResult:
    input_feature_names: tuple[str, ...]
    folds: tuple[LeaveOneWorkOutFoldResult, ...]
    summary: LeaveOneWorkOutSummary

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_feature_names", tuple(self.input_feature_names))
        object.__setattr__(self, "folds", tuple(self.folds))
