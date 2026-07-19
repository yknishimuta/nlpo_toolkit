from __future__ import annotations

from dataclasses import dataclass

from nlpo_toolkit.stylometry.errors import StylometryError
from nlpo_toolkit.stylometry.ports import AuthorshipMetadataReader
from nlpo_toolkit.stylometry.verification_evaluation_models import (
    VerificationEvaluationOutcome,
    VerificationExpectedClass,
    classify_verification_evaluation_outcome,
)

from ..ports import FeatureCommandDependencies
from .corpus_stylometry_support import prepare_corpus_stylometry_data
from .corpus_verification_evaluation_models import (
    CorpusVerificationEvaluationRequest,
    CorpusVerificationEvaluationResult,
    VerificationEvaluationFold,
    VerificationEvaluationSummary,
)
from .corpus_verification_service import evaluate_prepared_corpus_verification
from .errors import FeatureError


@dataclass(frozen=True)
class CorpusVerificationEvaluationDependencies:
    features: FeatureCommandDependencies
    read_metadata: AuthorshipMetadataReader


def build_verification_evaluation_summary(
    candidate_author: str, folds: tuple[VerificationEvaluationFold, ...]
) -> VerificationEvaluationSummary:
    genuine = sum(item.expected_class is VerificationExpectedClass.GENUINE for item in folds)
    impostor = len(folds) - genuine
    counts = {
        outcome: sum(item.outcome is outcome for item in folds)
        for outcome in VerificationEvaluationOutcome
    }
    correct = counts[VerificationEvaluationOutcome.CORRECT_ACCEPT] + counts[
        VerificationEvaluationOutcome.CORRECT_REJECT
    ]
    decisive = sum(item.is_decisive for item in folds)
    genuine_accept = counts[VerificationEvaluationOutcome.CORRECT_ACCEPT] / genuine
    impostor_reject = counts[VerificationEvaluationOutcome.CORRECT_REJECT] / impostor
    return VerificationEvaluationSummary(
        candidate_author,
        genuine,
        impostor,
        counts[VerificationEvaluationOutcome.CORRECT_ACCEPT],
        counts[VerificationEvaluationOutcome.FALSE_REJECT],
        counts[VerificationEvaluationOutcome.GENUINE_INCONCLUSIVE],
        counts[VerificationEvaluationOutcome.CORRECT_REJECT],
        counts[VerificationEvaluationOutcome.FALSE_ACCEPT],
        counts[VerificationEvaluationOutcome.IMPOSTOR_INCONCLUSIVE],
        genuine_accept,
        counts[VerificationEvaluationOutcome.FALSE_REJECT] / genuine,
        counts[VerificationEvaluationOutcome.GENUINE_INCONCLUSIVE] / genuine,
        impostor_reject,
        counts[VerificationEvaluationOutcome.FALSE_ACCEPT] / impostor,
        counts[VerificationEvaluationOutcome.IMPOSTOR_INCONCLUSIVE] / impostor,
        decisive / len(folds),
        correct / decisive if decisive else 0.0,
        correct / len(folds),
        (genuine_accept + impostor_reject) / 2.0,
    )


def execute_corpus_verification_evaluation(
    request: CorpusVerificationEvaluationRequest,
    *,
    dependencies: CorpusVerificationEvaluationDependencies,
) -> CorpusVerificationEvaluationResult:
    metadata = dependencies.read_metadata(
        request.metadata_path,
        input_format=request.metadata_format,
        id_column=request.metadata_group_column,
        author_column=request.author_column,
        work_column=request.work_column,
    )
    prepared = prepare_corpus_stylometry_data(
        request.features, metadata=metadata, dependencies=dependencies.features
    )
    ordered_works = tuple(
        dict.fromkeys(prepared.assignments[label][1] for label in prepared.labels)
    )
    authors = {
        work: prepared.assignments[label][0]
        for label in prepared.labels
        for work in (prepared.assignments[label][1],)
    }
    candidate_count = sum(author == request.candidate_author for author in authors.values())
    background_count = len(authors) - candidate_count
    if candidate_count < 4:
        raise FeatureError(
            "verification evaluation requires at least four candidate works; "
            f"author {request.candidate_author!r} has {candidate_count}"
        )
    if background_count < 3:
        raise FeatureError(
            "verification evaluation requires at least three background works; "
            f"found {background_count}"
        )
    folds = []
    for fold_index, query_work in enumerate(ordered_works, start=1):
        query_author = authors[query_work]
        try:
            corpus_result = evaluate_prepared_corpus_verification(
                prepared,
                candidate_author=request.candidate_author,
                query_work=query_work,
                thresholds=request.thresholds,
            )
        except (FeatureError, StylometryError) as exc:
            raise StylometryError(
                f"verification evaluation fold failed for query work {query_work!r}: {exc}"
            ) from exc
        expected = (
            VerificationExpectedClass.GENUINE
            if query_author == request.candidate_author
            else VerificationExpectedClass.IMPOSTOR
        )
        folds.append(
            VerificationEvaluationFold(
                fold_index,
                query_work,
                query_author,
                expected,
                classify_verification_evaluation_outcome(
                    expected, corpus_result.verification.decision
                ),
                corpus_result,
            )
        )
    fold_tuple = tuple(folds)
    return CorpusVerificationEvaluationResult(
        request.candidate_author,
        request.thresholds,
        fold_tuple,
        build_verification_evaluation_summary(request.candidate_author, fold_tuple),
    )
