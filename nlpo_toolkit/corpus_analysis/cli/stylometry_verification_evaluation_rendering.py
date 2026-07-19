from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import TextIO

from ..features.corpus_verification_evaluation_models import (
    CorpusVerificationEvaluationResult,
)
from .stylometry_verification_rendering import (
    CALIBRATION_COLUMNS,
    verification_calibration_rows,
)


FOLD_COLUMNS = (
    "fold_index", "query_work", "query_author", "expected_class", "decision",
    "outcome", "is_decisive", "is_correct", "query_sample_count",
    "query_distance", "genuine_boundary", "impostor_boundary",
    "accept_threshold", "reject_threshold", "candidate_reference_work_count",
    "background_work_count", "background_author_count", "input_feature_count",
    "retained_feature_count", "dropped_zero_variance_feature_count",
    "selected_feature_count", "selected_mfw_count",
    "selected_character_ngram_count", "selected_upos_ngram_count",
    "selected_morphology_count", "vocabulary_sha256", "nearest_background_work",
    "nearest_background_author", "nearest_background_distance",
    "candidate_vs_background_margin",
)


def write_verification_evaluation_folds(
    result: CorpusVerificationEvaluationResult,
    *,
    stream: TextIO,
    output_format: str,
) -> None:
    writer = csv.writer(stream, delimiter="," if output_format == "csv" else "\t")
    writer.writerow(FOLD_COLUMNS)
    for fold in result.folds:
        verification = fold.verification
        thresholds = verification.thresholds
        audit = fold.corpus_verification.vocabulary
        nearest = verification.nearest_background
        writer.writerow(
            (
                fold.fold_index, fold.query_work, fold.query_author,
                fold.expected_class.value, verification.decision.value,
                fold.outcome.value, "true" if fold.is_decisive else "false",
                "true" if fold.is_correct else "false", verification.query_sample_count,
                verification.query_distance, thresholds.genuine_boundary,
                thresholds.impostor_boundary, thresholds.accept_threshold,
                thresholds.reject_threshold, verification.candidate_reference_work_count,
                verification.background_work_count, verification.background_author_count,
                len(verification.input_feature_names), len(verification.retained_feature_names),
                len(verification.dropped_zero_variance_features),
                fold.corpus_verification.selected_feature_count,
                audit.selected_mfw_count, audit.selected_character_ngram_count,
                audit.selected_upos_ngram_count, audit.selected_morphology_count,
                audit.sha256, nearest.work_id, nearest.author, nearest.distance,
                nearest.candidate_vs_background_margin,
            )
        )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_verification_evaluation_summary(
    result: CorpusVerificationEvaluationResult, *, path: Path
) -> None:
    item = result.summary
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "method": "pseudo_unknown_work_verification_evaluation",
            "input_mode": "corpus",
            "metric": "burrows_delta",
            "candidate_author": result.candidate_author,
            "threshold_settings": {
                "genuine_quantile": result.thresholds.genuine_quantile,
                "impostor_quantile": result.thresholds.impostor_quantile,
            },
            "work_counts": {
                "genuine": item.genuine_work_count,
                "impostor": item.impostor_work_count,
                "total": item.genuine_work_count + item.impostor_work_count,
            },
            "outcome_counts": {
                "correct_accept": item.correct_accept_count,
                "false_reject": item.false_reject_count,
                "genuine_inconclusive": item.genuine_inconclusive_count,
                "correct_reject": item.correct_reject_count,
                "false_accept": item.false_accept_count,
                "impostor_inconclusive": item.impostor_inconclusive_count,
            },
            "rates": {
                "genuine_accept_rate": item.genuine_accept_rate,
                "false_reject_rate": item.false_reject_rate,
                "genuine_inconclusive_rate": item.genuine_inconclusive_rate,
                "impostor_reject_rate": item.impostor_reject_rate,
                "false_accept_rate": item.false_accept_rate,
                "impostor_inconclusive_rate": item.impostor_inconclusive_rate,
                "coverage": item.coverage,
                "decisive_accuracy": item.decisive_accuracy,
                "overall_correct_rate": item.overall_correct_rate,
                "balanced_correct_rate": item.balanced_correct_rate,
            },
            "limitations": {
                "pseudo_query_excluded_from_vocabulary_fit": True,
                "pseudo_query_excluded_from_zscore_fit": True,
                "pseudo_query_excluded_from_calibration": True,
                "thresholds_not_tuned_on_pseudo_queries": True,
                "inconclusive_reported_separately": True,
                "accept_does_not_prove_authenticity": True,
                "reject_does_not_prove_forgery": True,
            },
        },
    )


def _vocabulary_value(fold) -> dict[str, object]:
    audit = fold.corpus_verification.vocabulary
    return {
        "fold_index": fold.fold_index,
        "query_work": fold.query_work,
        "query_author": fold.query_author,
        "expected_class": fold.expected_class.value,
        "query_excluded": True,
        "fit_scope": audit.fit_scope,
        "selected_feature_count": fold.corpus_verification.selected_feature_count,
        "selected_mfw_count": audit.selected_mfw_count,
        "selected_character_ngram_count": audit.selected_character_ngram_count,
        "selected_upos_ngram_count": audit.selected_upos_ngram_count,
        "selected_morphology_count": audit.selected_morphology_count,
        "mfw_terms": list(audit.mfw_terms),
        "character_ngrams": [
            {"mode": term.mode.value, "size": term.size, "value": term.value, "column_name": term.column_name}
            for term in audit.character_ngrams
        ],
        "upos_ngrams": [
            {"size": term.size, "values": list(term.tags), "column_name": term.column_name}
            for term in audit.upos_ngrams
        ],
        "morphology": (
            {
                "attributes": list(audit.morphology.attributes),
                "values": [{"attribute": item.attribute, "value": item.value} for item in audit.morphology.values],
                "bundles": [[{"attribute": item.attribute, "value": item.value} for item in bundle.features] for bundle in audit.morphology.bundles],
            }
            if audit.morphology is not None else None
        ),
        "sha256": audit.sha256,
    }


def write_verification_evaluation_vocabulary_audit(
    result: CorpusVerificationEvaluationResult, *, path: Path
) -> None:
    _atomic_json(path, {"schema_version": 1, "method": "pseudo_unknown_work_verification_evaluation", "folds": [_vocabulary_value(fold) for fold in result.folds]})


def write_verification_evaluation_calibration(
    result: CorpusVerificationEvaluationResult, *, path: Path, output_format: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="," if output_format == "csv" else "\t")
        writer.writerow(("fold_index", "query_work", "query_author", "expected_class") + CALIBRATION_COLUMNS)
        for fold in result.folds:
            prefix = (fold.fold_index, fold.query_work, fold.query_author, fold.expected_class.value)
            writer.writerows(prefix + row for row in verification_calibration_rows(fold.verification))
