from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence

from .authorship import build_work_profiles
from .errors import StylometryError
from .evaluation_models import LabeledFeatureDataset, WorkProfile
from .stability_models import (
    ReferenceWorkRole,
    ResamplingAxis,
    VerificationStabilitySettings,
    VerificationStabilityStatus,
)
from .stability_resampling import (
    bootstrap_work_profiles,
    derive_iteration_seed,
    select_profile_features,
    stable_feature_hash,
    subsample_feature_names,
    subsample_reference_works,
)
from .stability_results import (
    DecisionStabilitySummary,
    FeatureStabilitySummary,
    NearestBackgroundFrequency,
    RejectedAttemptReason,
    ResamplingDistributionSummary,
    VerificationReplicate,
    VerificationStabilityResult,
    WorkInclusionSummary,
)
from .verification import evaluate_verification, linear_quantile
from .verification_models import VerificationDecision, VerificationThresholdSettings
from .verification_results import VerificationResult

ZERO_VARIANCE_REASON = "all_selected_features_zero_variance"


def _partition(
    profiles: Sequence[WorkProfile], candidate_author: str, query_work: str
) -> tuple[tuple[WorkProfile, ...], tuple[WorkProfile, ...], WorkProfile]:
    query = tuple(item for item in profiles if item.work_id == query_work)
    if len(query) != 1:
        raise StylometryError(f"query work not found: {query_work!r}")
    candidate = tuple(
        item
        for item in profiles
        if item.work_id != query_work and item.author == candidate_author
    )
    background = tuple(
        item
        for item in profiles
        if item.work_id != query_work and item.author != candidate_author
    )
    return candidate, background, query[0]


def validate_resampling_can_change(
    dataset: LabeledFeatureDataset,
    profiles: tuple[WorkProfile, ...],
    *,
    candidate_author: str,
    query_work: str,
    settings: VerificationStabilitySettings,
) -> None:
    candidate, background, _ = _partition(profiles, candidate_author, query_work)
    changes = False
    if ResamplingAxis.WORKS in settings.axes:
        candidate_size = min(
            len(candidate), max(3, math.ceil(len(candidate) * settings.work_fraction))
        )
        background_size = min(
            len(background), max(2, math.ceil(len(background) * settings.work_fraction))
        )
        changes |= candidate_size < len(candidate) or background_size < len(background)
    if ResamplingAxis.SAMPLES in settings.axes:
        counts = Counter(item.work_id for item in dataset.observations)
        changes |= any(count > 1 for count in counts.values())
    if ResamplingAxis.FEATURES in settings.axes:
        selected = max(1, math.ceil(dataset.feature_count * settings.feature_fraction))
        changes |= selected < dataset.feature_count
    if not changes:
        raise StylometryError(
            "resampling settings cannot change the verification dataset"
        )


def summarize_distribution(
    values: Sequence[float], lower: float, upper: float
) -> ResamplingDistributionSummary:
    items = tuple(float(value) for value in values)
    mean = sum(items) / len(items)
    variance = (
        0.0
        if len(items) == 1
        else sum((value - mean) ** 2 for value in items) / (len(items) - 1)
    )
    return ResamplingDistributionSummary(
        len(items),
        min(items),
        linear_quantile(items, lower),
        linear_quantile(items, 0.5),
        linear_quantile(items, upper),
        max(items),
        mean,
        math.sqrt(variance),
    )


def summarize_decisions(
    decisions: Sequence[VerificationDecision],
    *,
    base_decision: VerificationDecision,
    stability_threshold: float,
) -> DecisionStabilitySummary:
    counts = Counter(decisions)
    priority = (
        VerificationDecision.INCONCLUSIVE,
        VerificationDecision.ACCEPT,
        VerificationDecision.REJECT,
    )
    modal = max(
        priority,
        key=lambda decision: (counts[decision], -priority.index(decision)),
    )
    modal_count = counts[modal]
    status = (
        VerificationStabilityStatus.STABLE
        if modal_count / len(decisions) >= stability_threshold
        else VerificationStabilityStatus.UNSTABLE
    )
    return DecisionStabilitySummary(
        status,
        modal,
        modal_count,
        sum(item is base_decision for item in decisions),
        counts[VerificationDecision.ACCEPT],
        counts[VerificationDecision.INCONCLUSIVE],
        counts[VerificationDecision.REJECT],
        len(decisions),
    )


def _resampled_profiles(
    dataset: LabeledFeatureDataset,
    base_profiles: tuple[WorkProfile, ...],
    included_ids: frozenset[str],
    *,
    sample_bootstrap: bool,
    rng: random.Random,
) -> tuple[WorkProfile, ...]:
    return (
        bootstrap_work_profiles(dataset, included_work_ids=included_ids, rng=rng)
        if sample_bootstrap
        else tuple(item for item in base_profiles if item.work_id in included_ids)
    )


def evaluate_verification_stability(
    dataset: LabeledFeatureDataset,
    *,
    candidate_author: str,
    query_work: str,
    verification_thresholds: VerificationThresholdSettings,
    settings: VerificationStabilitySettings,
) -> VerificationStabilityResult:
    base_profiles = build_work_profiles(dataset)
    base = evaluate_verification(
        dataset.feature_names,
        base_profiles,
        candidate_author=candidate_author,
        query_work=query_work,
        settings=verification_thresholds,
    )
    validate_resampling_can_change(
        dataset,
        base_profiles,
        candidate_author=candidate_author,
        query_work=query_work,
        settings=settings,
    )
    all_candidate, all_background, query = _partition(
        base_profiles, candidate_author, query_work
    )
    replicates: list[VerificationReplicate] = []
    attempts = 0
    rejected: Counter[str] = Counter()
    while len(replicates) < settings.iterations and attempts < settings.max_attempts:
        attempts += 1
        iteration = len(replicates) + 1
        seed = derive_iteration_seed(settings.seed, iteration, attempts)
        rng = random.Random(seed)
        candidate = all_candidate
        background = all_background
        if ResamplingAxis.WORKS in settings.axes:
            candidate = subsample_reference_works(
                candidate, fraction=settings.work_fraction, minimum=3, rng=rng
            )
            background = subsample_reference_works(
                background, fraction=settings.work_fraction, minimum=2, rng=rng
            )
        included_ids = frozenset(
            item.work_id for item in candidate + background
        ) | frozenset((query.work_id,))
        profiles = _resampled_profiles(
            dataset,
            base_profiles,
            included_ids,
            sample_bootstrap=ResamplingAxis.SAMPLES in settings.axes,
            rng=rng,
        )
        selected_names = dataset.feature_names
        if ResamplingAxis.FEATURES in settings.axes:
            selected_names = subsample_feature_names(
                dataset.feature_names, fraction=settings.feature_fraction, rng=rng
            )
        profiles = select_profile_features(
            profiles,
            input_feature_names=dataset.feature_names,
            selected_feature_names=selected_names,
        )
        try:
            result = evaluate_verification(
                selected_names,
                profiles,
                candidate_author=candidate_author,
                query_work=query_work,
                settings=verification_thresholds,
            )
        except StylometryError as exc:
            if str(exc) == (
                "all selected features have zero variance in verification reference works"
            ):
                rejected[ZERO_VARIANCE_REASON] += 1
                continue
            raise
        candidate_ids = tuple(sorted(item.work_id for item in candidate))
        background_ids = tuple(sorted(item.work_id for item in background))
        replicates.append(
            VerificationReplicate(
                iteration,
                attempts,
                seed,
                result,
                candidate_ids,
                background_ids,
                selected_names,
                stable_feature_hash(selected_names),
                stable_feature_hash(result.retained_feature_names),
            )
        )
    if len(replicates) < settings.iterations:
        details = ", ".join(f"{key}={value}" for key, value in sorted(rejected.items()))
        raise StylometryError(
            f"verification stability reached max attempts: {len(replicates)} "
            f"successful / {attempts} attempts; rejected: {details or 'none'}"
        )
    completed = tuple(replicates)
    extractors: tuple[tuple[str, Callable[[VerificationResult], float]], ...] = (
        ("query_distance", lambda value: value.query_distance),
        ("genuine_boundary", lambda value: value.thresholds.genuine_boundary),
        ("impostor_boundary", lambda value: value.thresholds.impostor_boundary),
        ("accept_threshold", lambda value: value.thresholds.accept_threshold),
        ("reject_threshold", lambda value: value.thresholds.reject_threshold),
        (
            "nearest_background_distance",
            lambda value: value.nearest_background.distance,
        ),
        (
            "candidate_vs_background_margin",
            lambda value: value.nearest_background.candidate_vs_background_margin,
        ),
        (
            "retained_feature_count",
            lambda value: float(len(value.retained_feature_names)),
        ),
        (
            "candidate_reference_work_count",
            lambda value: float(value.candidate_reference_work_count),
        ),
        ("background_work_count", lambda value: float(value.background_work_count)),
    )
    distributions = tuple(
        (
            name,
            summarize_distribution(
                tuple(extractor(item.result) for item in completed),
                settings.interval.lower,
                settings.interval.upper,
            ),
        )
        for name, extractor in extractors
    )
    work_rows = _work_inclusion(completed, all_candidate, all_background)
    nearest_rows = _nearest_frequency(completed, all_background)
    feature_rows = tuple(
        FeatureStabilitySummary(
            feature,
            sum(feature in item.selected_feature_names for item in completed),
            sum(feature in item.result.retained_feature_names for item in completed),
            len(completed),
        )
        for feature in dataset.feature_names
    )
    return VerificationStabilityResult(
        base,
        settings,
        completed,
        attempts,
        tuple(
            RejectedAttemptReason(reason, count)
            for reason, count in sorted(rejected.items())
        ),
        summarize_decisions(
            tuple(item.result.decision for item in completed),
            base_decision=base.decision,
            stability_threshold=settings.stability_threshold,
        ),
        distributions,
        work_rows,
        nearest_rows,
        feature_rows,
    )


def _work_inclusion(
    replicates: tuple[VerificationReplicate, ...],
    candidate: tuple[WorkProfile, ...],
    background: tuple[WorkProfile, ...],
) -> tuple[WorkInclusionSummary, ...]:
    rows = []
    for role, works in (
        (ReferenceWorkRole.CANDIDATE, candidate),
        (ReferenceWorkRole.BACKGROUND, background),
    ):
        for work in sorted(works, key=lambda item: (item.author, item.work_id)):
            included = sum(
                work.work_id
                in (
                    item.candidate_reference_works
                    if role is ReferenceWorkRole.CANDIDATE
                    else item.background_works
                )
                for item in replicates
            )
            rows.append(
                WorkInclusionSummary(
                    work.author, work.work_id, role, len(replicates), included
                )
            )
    return tuple(rows)


def _nearest_frequency(
    replicates: tuple[VerificationReplicate, ...],
    background: tuple[WorkProfile, ...],
) -> tuple[NearestBackgroundFrequency, ...]:
    return tuple(
        NearestBackgroundFrequency(
            work.author,
            work.work_id,
            sum(
                item.result.nearest_background.work_id == work.work_id
                for item in replicates
            ),
            sum(work.work_id in item.background_works for item in replicates),
            len(replicates),
        )
        for work in sorted(background, key=lambda item: (item.author, item.work_id))
    )
