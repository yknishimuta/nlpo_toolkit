from __future__ import annotations

from .delta import burrows_delta
from .errors import StylometryError
from .evaluation_models import AuthorProfile, LeaveOneWorkOutFold, WorkProfile
from .evaluation_results import (
    AuthorCandidateDistance,
    AuthorEvaluationSummary,
    LeaveOneWorkOutEvaluationResult,
    LeaveOneWorkOutFoldResult,
    LeaveOneWorkOutSummary,
)
from .models import (
    FeatureDataset,
    FeatureObservation,
    StandardizedFeatureDataset,
    StandardizedObservation,
)
from .standardization import fit_zscore_model, transform_feature_dataset


def build_leave_one_work_out_folds(
    profiles: tuple[WorkProfile, ...],
) -> tuple[LeaveOneWorkOutFold, ...]:
    return tuple(
        LeaveOneWorkOutFold(
            fold_index=index + 1,
            test_work=test,
            training_works=profiles[:index] + profiles[index + 1 :],
        )
        for index, test in enumerate(profiles)
    )


def work_feature_dataset(
    feature_names: tuple[str, ...], profiles: tuple[WorkProfile, ...]
) -> FeatureDataset:
    return FeatureDataset(
        feature_names,
        tuple(FeatureObservation(item.work_id, item.values) for item in profiles),
    )


def build_author_profiles(
    training_works: tuple[WorkProfile, ...],
    standardized: StandardizedFeatureDataset,
) -> tuple[AuthorProfile, ...]:
    by_id = {item.identifier: item for item in standardized.observations}
    grouped: dict[str, list[WorkProfile]] = {}
    for work in training_works:
        grouped.setdefault(work.author, []).append(work)
    return tuple(
        AuthorProfile(
            author=author,
            training_work_ids=tuple(work.work_id for work in works),
            values=tuple(
                sum(by_id[work.work_id].values[index] for work in works) / len(works)
                for index in range(len(standardized.feature_names))
            ),
        )
        for author, works in grouped.items()
    )


def evaluate_lowo(
    feature_names: tuple[str, ...], profiles: tuple[WorkProfile, ...]
) -> LeaveOneWorkOutEvaluationResult:
    results = []
    for fold in build_leave_one_work_out_folds(profiles):
        results.append(
            evaluate_lowo_fold(
                feature_names,
                training_work_profiles=fold.training_works,
                test_work_profile=fold.test_work,
                fold_index=fold.fold_index,
            )
        )
    folds = tuple(results)
    author_order: list[str] = []
    for fold in folds:
        if fold.actual_author not in author_order:
            author_order.append(fold.actual_author)
    per_author = tuple(
        AuthorEvaluationSummary(
            author=author,
            work_count=sum(fold.actual_author == author for fold in folds),
            correct_work_count=sum(
                fold.actual_author == author and fold.is_correct for fold in folds
            ),
        )
        for author in author_order
    )
    summary = LeaveOneWorkOutSummary(per_author=per_author, folds=folds)
    return LeaveOneWorkOutEvaluationResult(feature_names, folds, summary)


def evaluate_lowo_fold(
    feature_names: tuple[str, ...],
    *,
    training_work_profiles: tuple[WorkProfile, ...],
    test_work_profile: WorkProfile,
    fold_index: int,
) -> LeaveOneWorkOutFoldResult:
    training_dataset = work_feature_dataset(feature_names, training_work_profiles)
    try:
        model = fit_zscore_model(training_dataset)
    except StylometryError as exc:
        if "all selected features have zero variance" in str(exc):
            raise StylometryError(
                "all selected features have zero variance in training fold "
                f"for held-out work {test_work_profile.work_id!r}"
            ) from exc
        raise
    standardized_training = transform_feature_dataset(training_dataset, model=model)
    test_dataset = work_feature_dataset(feature_names, (test_work_profile,))
    test_observation = transform_feature_dataset(
        test_dataset, model=model
    ).observations[0]
    author_profiles = build_author_profiles(
        training_work_profiles, standardized_training
    )
    candidates = tuple(
        sorted(
            (
                AuthorCandidateDistance(
                    author=profile.author,
                    distance=burrows_delta(
                        test_observation,
                        StandardizedObservation(profile.author, profile.values),
                    ),
                )
                for profile in author_profiles
            ),
            key=lambda item: (item.distance, item.author),
        )
    )
    return LeaveOneWorkOutFoldResult(
        fold_index=fold_index,
        work_id=test_work_profile.work_id,
        actual_author=test_work_profile.author,
        test_observation_ids=test_work_profile.observation_ids,
        training_work_ids=tuple(work.work_id for work in training_work_profiles),
        retained_feature_names=model.retained_feature_names,
        dropped_zero_variance_features=model.dropped_zero_variance_features,
        candidates=candidates,
    )
