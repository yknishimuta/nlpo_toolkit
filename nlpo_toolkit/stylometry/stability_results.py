from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import StylometryError
from .stability_models import (
    ReferenceWorkRole,
    VerificationStabilitySettings,
    VerificationStabilityStatus,
)
from .verification_models import VerificationDecision
from .verification_results import VerificationResult


@dataclass(frozen=True)
class VerificationReplicate:
    iteration: int
    attempt: int
    iteration_seed: int
    result: VerificationResult
    candidate_reference_works: tuple[str, ...]
    background_works: tuple[str, ...]
    selected_feature_names: tuple[str, ...]
    selected_features_sha256: str
    retained_features_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "candidate_reference_works",
            "background_works",
            "selected_feature_names",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class ResamplingDistributionSummary:
    count: int
    minimum: float
    lower_interval: float
    median: float
    upper_interval: float
    maximum: float
    mean: float
    sample_standard_deviation: float

    def __post_init__(self) -> None:
        values = (
            self.minimum,
            self.lower_interval,
            self.median,
            self.upper_interval,
            self.maximum,
            self.mean,
            self.sample_standard_deviation,
        )
        if self.count <= 0 or any(
            isinstance(v, bool) or not math.isfinite(v) for v in values
        ):
            raise StylometryError("resampling distribution values must be finite")


@dataclass(frozen=True)
class DecisionStabilitySummary:
    status: VerificationStabilityStatus
    modal_decision: VerificationDecision
    modal_decision_count: int
    base_decision_agreement_count: int
    accept_count: int
    inconclusive_count: int
    reject_count: int
    iteration_count: int

    @property
    def modal_decision_rate(self) -> float:
        return self.modal_decision_count / self.iteration_count

    @property
    def base_decision_agreement_rate(self) -> float:
        return self.base_decision_agreement_count / self.iteration_count

    @property
    def accept_rate(self) -> float:
        return self.accept_count / self.iteration_count

    @property
    def inconclusive_rate(self) -> float:
        return self.inconclusive_count / self.iteration_count

    @property
    def reject_rate(self) -> float:
        return self.reject_count / self.iteration_count


@dataclass(frozen=True)
class WorkInclusionSummary:
    author: str
    work_id: str
    role: ReferenceWorkRole
    available_iterations: int
    included_count: int

    def __post_init__(self) -> None:
        if not 0 <= self.included_count <= self.available_iterations:
            raise StylometryError("work inclusion counts are inconsistent")

    @property
    def included_rate(self) -> float:
        return self.included_count / self.available_iterations


@dataclass(frozen=True)
class NearestBackgroundFrequency:
    author: str
    work_id: str
    nearest_count: int
    included_count: int
    iteration_count: int

    def __post_init__(self) -> None:
        if not 0 <= self.nearest_count <= self.included_count <= self.iteration_count:
            raise StylometryError("nearest-background counts are inconsistent")

    @property
    def nearest_rate(self) -> float:
        return self.nearest_count / self.iteration_count

    @property
    def nearest_given_included_rate(self) -> float | None:
        return self.nearest_count / self.included_count if self.included_count else None


@dataclass(frozen=True)
class FeatureStabilitySummary:
    feature: str
    selected_count: int
    retained_count: int
    iteration_count: int

    def __post_init__(self) -> None:
        if not self.feature or not (
            0 <= self.retained_count <= self.selected_count <= self.iteration_count
        ):
            raise StylometryError("feature stability counts are inconsistent")

    @property
    def selected_rate(self) -> float:
        return self.selected_count / self.iteration_count

    @property
    def retained_rate(self) -> float:
        return self.retained_count / self.iteration_count

    @property
    def retained_given_selected_rate(self) -> float | None:
        return (
            self.retained_count / self.selected_count if self.selected_count else None
        )


@dataclass(frozen=True)
class RejectedAttemptReason:
    reason: str
    count: int


@dataclass(frozen=True)
class VerificationStabilityResult:
    base_result: VerificationResult
    settings: VerificationStabilitySettings
    replicates: tuple[VerificationReplicate, ...]
    attempted_iterations: int
    rejected_attempt_reasons: tuple[RejectedAttemptReason, ...]
    decision_stability: DecisionStabilitySummary
    distributions: tuple[tuple[str, ResamplingDistributionSummary], ...]
    work_inclusion: tuple[WorkInclusionSummary, ...]
    nearest_background_frequency: tuple[NearestBackgroundFrequency, ...]
    feature_stability: tuple[FeatureStabilitySummary, ...]

    def __post_init__(self) -> None:
        for name in (
            "replicates",
            "rejected_attempt_reasons",
            "distributions",
            "work_inclusion",
            "nearest_background_frequency",
            "feature_stability",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def successful_iterations(self) -> int:
        return len(self.replicates)

    @property
    def rejected_attempts(self) -> int:
        return self.attempted_iterations - self.successful_iterations
