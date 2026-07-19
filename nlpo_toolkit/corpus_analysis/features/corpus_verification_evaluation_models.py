from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from nlpo_toolkit.stylometry.models import InputFormat
from nlpo_toolkit.stylometry.verification_evaluation_models import (
    VerificationEvaluationOutcome,
    VerificationExpectedClass,
)
from nlpo_toolkit.stylometry.verification_models import (
    VerificationDecision,
    VerificationThresholdSettings,
)
from nlpo_toolkit.stylometry.verification_results import VerificationResult

from ..requests import CorpusPreparationRequest
from .corpus_verification_models import CorpusVerificationResult
from .models import FeatureRequest


@dataclass(frozen=True)
class CorpusVerificationEvaluationRequest:
    features: FeatureRequest
    metadata_path: Path
    candidate_author: str
    thresholds: VerificationThresholdSettings = VerificationThresholdSettings()
    metadata_format: InputFormat = "csv"
    metadata_group_column: str = "group"
    author_column: str = "author"
    work_column: str = "work"

    def __post_init__(self) -> None:
        if not isinstance(self.metadata_path, Path):
            raise ValueError("metadata path must be a Path")
        if self.metadata_format not in ("csv", "tsv"):
            raise ValueError("metadata format must be 'csv' or 'tsv'")
        for value, name in (
            (self.candidate_author, "candidate author"),
            (self.metadata_group_column, "metadata group column"),
            (self.author_column, "author column"),
            (self.work_column, "work column"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")

    @property
    def corpus(self) -> CorpusPreparationRequest:
        return self.features.corpus


@dataclass(frozen=True)
class VerificationEvaluationFold:
    fold_index: int
    query_work: str
    query_author: str
    expected_class: VerificationExpectedClass
    outcome: VerificationEvaluationOutcome
    corpus_verification: CorpusVerificationResult

    def __post_init__(self) -> None:
        if self.fold_index < 1:
            raise ValueError("verification evaluation fold index must be positive")
        if not self.query_work.strip() or not self.query_author.strip():
            raise ValueError("verification evaluation query labels must not be empty")
        if self.verification.query_work != self.query_work:
            raise ValueError("verification fold query work does not match result")
        if self.corpus_verification.vocabulary.query_work != self.query_work:
            raise ValueError("verification fold query work does not match vocabulary")

    @property
    def verification(self) -> VerificationResult:
        return self.corpus_verification.verification

    @property
    def is_decisive(self) -> bool:
        return self.verification.decision is not VerificationDecision.INCONCLUSIVE

    @property
    def is_correct(self) -> bool:
        return self.outcome in {
            VerificationEvaluationOutcome.CORRECT_ACCEPT,
            VerificationEvaluationOutcome.CORRECT_REJECT,
        }


@dataclass(frozen=True)
class VerificationEvaluationSummary:
    candidate_author: str
    genuine_work_count: int
    impostor_work_count: int
    correct_accept_count: int
    false_reject_count: int
    genuine_inconclusive_count: int
    correct_reject_count: int
    false_accept_count: int
    impostor_inconclusive_count: int
    genuine_accept_rate: float
    false_reject_rate: float
    genuine_inconclusive_rate: float
    impostor_reject_rate: float
    false_accept_rate: float
    impostor_inconclusive_rate: float
    coverage: float
    decisive_accuracy: float
    overall_correct_rate: float
    balanced_correct_rate: float

    def __post_init__(self) -> None:
        if not self.candidate_author.strip():
            raise ValueError("candidate author must not be empty")
        counts = (
            self.genuine_work_count,
            self.impostor_work_count,
            self.correct_accept_count,
            self.false_reject_count,
            self.genuine_inconclusive_count,
            self.correct_reject_count,
            self.false_accept_count,
            self.impostor_inconclusive_count,
        )
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts):
            raise ValueError("verification evaluation counts must be non-negative integers")
        rates = (
            self.genuine_accept_rate,
            self.false_reject_rate,
            self.genuine_inconclusive_rate,
            self.impostor_reject_rate,
            self.false_accept_rate,
            self.impostor_inconclusive_rate,
            self.coverage,
            self.decisive_accuracy,
            self.overall_correct_rate,
            self.balanced_correct_rate,
        )
        if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in rates):
            raise ValueError("verification evaluation rates must be finite probabilities")


@dataclass(frozen=True)
class CorpusVerificationEvaluationResult:
    candidate_author: str
    thresholds: VerificationThresholdSettings
    folds: tuple[VerificationEvaluationFold, ...]
    summary: VerificationEvaluationSummary

    def __post_init__(self) -> None:
        object.__setattr__(self, "folds", tuple(self.folds))
        if not self.folds:
            raise ValueError("verification evaluation requires folds")
        if tuple(item.fold_index for item in self.folds) != tuple(range(1, len(self.folds) + 1)):
            raise ValueError("verification evaluation fold indexes must be consecutive")
        works = tuple(item.query_work for item in self.folds)
        if len(works) != len(set(works)):
            raise ValueError("verification evaluation query works must be unique")
