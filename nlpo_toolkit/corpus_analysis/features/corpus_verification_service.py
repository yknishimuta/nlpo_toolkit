from __future__ import annotations

from dataclasses import dataclass

from nlpo_toolkit.stylometry.authorship import build_work_profiles
from nlpo_toolkit.stylometry.ports import AuthorshipMetadataReader
from nlpo_toolkit.stylometry.verification import evaluate_verification
from nlpo_toolkit.stylometry.verification_models import VerificationThresholdSettings

from ..ports import FeatureCommandDependencies
from .corpus_stylometry_support import (
    PreparedCorpusStylometryData,
    build_labeled_feature_dataset,
    prepare_corpus_stylometry_data,
)
from .corpus_verification_models import (
    CorpusVerificationRequest,
    CorpusVerificationResult,
    CorpusVerificationVocabularyAudit,
)
from .engine import build_feature_rows, fit_feature_vocabulary
from .errors import FeatureError


@dataclass(frozen=True)
class CorpusVerificationDependencies:
    features: FeatureCommandDependencies
    read_metadata: AuthorshipMetadataReader


def execute_corpus_verification(
    request: CorpusVerificationRequest,
    *,
    dependencies: CorpusVerificationDependencies,
) -> CorpusVerificationResult:
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
    return evaluate_prepared_corpus_verification(
        prepared,
        candidate_author=request.candidate_author,
        query_work=request.query_work,
        thresholds=request.thresholds,
    )


def evaluate_prepared_corpus_verification(
    prepared: PreparedCorpusStylometryData,
    *,
    candidate_author: str,
    query_work: str,
    thresholds: VerificationThresholdSettings,
) -> CorpusVerificationResult:
    reference_corpora = tuple(
        corpus
        for corpus in prepared.analyzed
        if prepared.assignments[corpus.source.label][1] != query_work
    )
    query_corpora = tuple(
        corpus
        for corpus in prepared.analyzed
        if prepared.assignments[corpus.source.label][1] == query_work
    )
    if not query_corpora:
        raise FeatureError(f"query work not found in prepared corpora: {query_work!r}")
    if not reference_corpora:
        raise FeatureError("verification reference corpus is empty after excluding query work")
    vocabulary = fit_feature_vocabulary(reference_corpora, options=prepared.options)
    rows = build_feature_rows(
        prepared.analyzed,
        options=prepared.options,
        vocabulary=vocabulary,
    )
    labeled = build_labeled_feature_dataset(rows, prepared.assignments)
    verification = evaluate_verification(
        labeled.feature_names,
        build_work_profiles(labeled),
        candidate_author=candidate_author,
        query_work=query_work,
        settings=thresholds,
    )
    audit = CorpusVerificationVocabularyAudit(
        query_work,
        vocabulary.mfw_terms,
        vocabulary.character_ngrams.terms if vocabulary.character_ngrams else (),
        vocabulary.upos_ngrams.terms if vocabulary.upos_ngrams else (),
        vocabulary.morphology,
    )
    return CorpusVerificationResult(verification, len(labeled.feature_names), audit)
