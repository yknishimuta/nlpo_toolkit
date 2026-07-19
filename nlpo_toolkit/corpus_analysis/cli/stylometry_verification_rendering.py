from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TextIO

from nlpo_toolkit.stylometry.verification_results import (
    VerificationDistributionSummary,
    VerificationResult,
)


CALIBRATION_COLUMNS = (
    "kind",
    "work",
    "author",
    "distance",
    "centroid_work_count",
    "centroid_works",
)


def verification_json_value(result: VerificationResult) -> dict[str, object]:
    nearest = result.nearest_background
    thresholds = result.thresholds
    genuine = result.genuine_distribution
    impostor = result.impostor_distribution
    return {
        "schema_version": 1,
        "method": "candidate_authorship_verification",
        "metric": "burrows_delta",
        "decision_target": "candidate_authorship",
        "decision": result.decision.value,
        "candidate_author": result.candidate_author,
        "query_work": result.query_work,
        "query_sample_count": result.query_sample_count,
        "query_distance": result.query_distance,
        "candidate_reference_work_count": result.candidate_reference_work_count,
        "background_work_count": result.background_work_count,
        "background_author_count": result.background_author_count,
        "reference_work_count": result.reference_work_count,
        "input_feature_count": len(result.input_feature_names),
        "retained_feature_count": len(result.retained_feature_names),
        "dropped_zero_variance_count": len(result.dropped_zero_variance_features),
        "dropped_zero_variance_features": list(result.dropped_zero_variance_features),
        "candidate_reference_works": list(result.candidate_reference_works),
        "thresholds": {
            "genuine_quantile": thresholds.genuine_quantile,
            "impostor_quantile": thresholds.impostor_quantile,
            "genuine_boundary": thresholds.genuine_boundary,
            "impostor_boundary": thresholds.impostor_boundary,
            "accept_threshold": thresholds.accept_threshold,
            "reject_threshold": thresholds.reject_threshold,
        },
        "genuine_distribution": _distribution_value(genuine),
        "impostor_distribution": _distribution_value(impostor),
        "nearest_background": {
            "work": nearest.work_id,
            "author": nearest.author,
            "distance": nearest.distance,
            "candidate_vs_background_margin": nearest.candidate_vs_background_margin,
        },
        "limitations": {
            "closed_feature_space": True,
            "query_excluded_from_standardization": True,
            "query_excluded_from_threshold_calibration": True,
            "authenticity_not_proven": True,
        },
    }


def _distribution_value(
    summary: VerificationDistributionSummary,
) -> dict[str, object]:
    return {
        "count": summary.count,
        "minimum": summary.minimum,
        "median": summary.median,
        "maximum": summary.maximum,
        "selected_quantile": summary.selected_quantile,
        "selected_quantile_value": summary.selected_quantile_value,
    }


def write_verification_json(result: VerificationResult, *, stream: TextIO) -> None:
    json.dump(
        verification_json_value(result), stream,
        ensure_ascii=False, allow_nan=False, indent=2,
    )
    stream.write("\n")


def write_verification_calibration(
    result: VerificationResult, *, path: Path, output_format: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="," if output_format == "csv" else "\t")
        writer.writerow(CALIBRATION_COLUMNS)
        writer.writerows(verification_calibration_rows(result))


def verification_calibration_rows(
    result: VerificationResult,
) -> tuple[tuple[str | int | float, ...], ...]:
    return tuple(
        (
            score.kind.value,
            score.work_id,
            score.author,
            score.distance,
            len(score.centroid_work_ids),
            "|".join(score.centroid_work_ids),
        )
        for score in result.calibration_scores
    )
