from __future__ import annotations

from dataclasses import dataclass

from nlpo_toolkit.stylometry.authorship import (
    build_work_profiles,
    validate_lowo_dataset,
)
from nlpo_toolkit.stylometry.evaluation import evaluate_lowo_fold
from nlpo_toolkit.stylometry.evaluation_results import (
    AuthorEvaluationSummary,
    LeaveOneWorkOutSummary,
)
from nlpo_toolkit.stylometry.ports import AuthorshipMetadataReader

from ..ports import FeatureCommandDependencies
from .corpus_lowo_models import (
    CorpusLowoFoldResult,
    CorpusLowoRequest,
    CorpusLowoResult,
    FoldVocabularyAudit,
)
from .corpus_stylometry_support import (
    build_labeled_feature_dataset,
    prepare_corpus_stylometry_data,
)
from .engine import build_feature_rows, fit_feature_vocabulary


@dataclass(frozen=True)
class CorpusLowoDependencies:
    features: FeatureCommandDependencies
    read_metadata: AuthorshipMetadataReader


def execute_corpus_lowo(
    request: CorpusLowoRequest, *, dependencies: CorpusLowoDependencies
) -> CorpusLowoResult:
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
    labels = prepared.labels
    assignments = prepared.assignments
    options = prepared.options
    analyzed = prepared.analyzed
    work_order = []
    for label in labels:
        work = assignments[label][1]
        if work not in work_order:
            work_order.append(work)
    folds = []
    for fold_index, test_work in enumerate(work_order, start=1):
        training = tuple(
            corpus
            for corpus in analyzed
            if assignments[corpus.source.label][1] != test_work
        )
        testing = tuple(
            corpus
            for corpus in analyzed
            if assignments[corpus.source.label][1] == test_work
        )
        vocabulary = fit_feature_vocabulary(training, options=options)
        rows = build_feature_rows(
            training + testing, options=options, vocabulary=vocabulary
        )
        labeled = build_labeled_feature_dataset(rows, assignments)
        validate_lowo_dataset(labeled)
        profiles = build_work_profiles(labeled)
        test_profile = next(
            profile for profile in profiles if profile.work_id == test_work
        )
        training_profiles = tuple(
            profile for profile in profiles if profile.work_id != test_work
        )
        evaluation = evaluate_lowo_fold(
            labeled.feature_names,
            training_work_profiles=training_profiles,
            test_work_profile=test_profile,
            fold_index=fold_index,
        )
        audit = FoldVocabularyAudit(
            fold_index,
            test_work,
            vocabulary.mfw_terms,
            vocabulary.character_ngrams.terms if vocabulary.character_ngrams else (),
            vocabulary.upos_ngrams.terms if vocabulary.upos_ngrams else (),
            vocabulary.morphology,
        )
        folds.append(
            CorpusLowoFoldResult(evaluation, len(labeled.feature_names), audit)
        )
    base_folds = tuple(item.evaluation for item in folds)
    author_order = []
    for fold in base_folds:
        if fold.actual_author not in author_order:
            author_order.append(fold.actual_author)
    per_author = tuple(
        AuthorEvaluationSummary(
            author,
            sum(fold.actual_author == author for fold in base_folds),
            sum(
                fold.actual_author == author and fold.is_correct for fold in base_folds
            ),
        )
        for author in author_order
    )
    summary = LeaveOneWorkOutSummary(per_author, base_folds)
    return CorpusLowoResult(tuple(folds), summary)
