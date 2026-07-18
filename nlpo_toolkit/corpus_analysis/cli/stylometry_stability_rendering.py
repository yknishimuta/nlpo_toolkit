from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import TextIO

from nlpo_toolkit.stylometry.stability_results import (
    ResamplingDistributionSummary,
    VerificationStabilityResult,
)

REPLICATE_COLUMNS = (
    "iteration",
    "attempt",
    "iteration_seed",
    "decision",
    "agrees_with_base",
    "query_distance",
    "genuine_boundary",
    "impostor_boundary",
    "accept_threshold",
    "reject_threshold",
    "candidate_reference_work_count",
    "background_work_count",
    "selected_feature_count",
    "retained_feature_count",
    "dropped_zero_variance_count",
    "nearest_background_author",
    "nearest_background_work",
    "nearest_background_distance",
    "candidate_vs_background_margin",
    "candidate_reference_works",
    "background_works",
    "selected_features_sha256",
    "retained_features_sha256",
)

FEATURE_COLUMNS = (
    "feature",
    "selected_count",
    "selected_rate",
    "retained_count",
    "retained_rate",
    "retained_given_selected_rate",
)


def _distribution_value(value: ResamplingDistributionSummary) -> dict[str, object]:
    return {
        "count": value.count,
        "minimum": value.minimum,
        "lower_interval": value.lower_interval,
        "median": value.median,
        "upper_interval": value.upper_interval,
        "maximum": value.maximum,
        "mean": value.mean,
        "sample_standard_deviation": value.sample_standard_deviation,
    }


def stability_json_value(result: VerificationStabilityResult) -> dict[str, object]:
    base = result.base_result
    decision = result.decision_stability
    settings = result.settings
    return {
        "schema_version": 1,
        "method": "candidate_authorship_verification_stability",
        "metric": "burrows_delta",
        "candidate_author": base.candidate_author,
        "query_work": base.query_work,
        "base_result": {
            "decision": base.decision.value,
            "query_distance": base.query_distance,
            "genuine_boundary": base.thresholds.genuine_boundary,
            "impostor_boundary": base.thresholds.impostor_boundary,
            "accept_threshold": base.thresholds.accept_threshold,
            "reject_threshold": base.thresholds.reject_threshold,
            "nearest_background_author": base.nearest_background.author,
            "nearest_background_work": base.nearest_background.work_id,
            "nearest_background_distance": base.nearest_background.distance,
        },
        "resampling": {
            "axes": [axis.value for axis in settings.axes],
            "iterations": settings.iterations,
            "seed": settings.seed,
            "max_attempts": settings.max_attempts,
            "work_fraction": settings.work_fraction,
            "feature_fraction": settings.feature_fraction,
            "stability_threshold": settings.stability_threshold,
            "interval_lower": settings.interval.lower,
            "interval_upper": settings.interval.upper,
            "successful_iterations": result.successful_iterations,
            "attempted_iterations": result.attempted_iterations,
            "rejected_attempts": result.rejected_attempts,
            "rejected_attempt_reasons": [
                {"reason": item.reason, "count": item.count}
                for item in result.rejected_attempt_reasons
            ],
        },
        "decision_stability": {
            "status": decision.status.value,
            "modal_decision": decision.modal_decision.value,
            "modal_decision_count": decision.modal_decision_count,
            "modal_decision_rate": decision.modal_decision_rate,
            "base_decision_agreement_count": decision.base_decision_agreement_count,
            "base_decision_agreement_rate": decision.base_decision_agreement_rate,
            "accept_count": decision.accept_count,
            "accept_rate": decision.accept_rate,
            "inconclusive_count": decision.inconclusive_count,
            "inconclusive_rate": decision.inconclusive_rate,
            "reject_count": decision.reject_count,
            "reject_rate": decision.reject_rate,
        },
        "distributions": {
            name: _distribution_value(summary) for name, summary in result.distributions
        },
        "work_inclusion": [
            {
                "author": item.author,
                "work": item.work_id,
                "role": item.role.value,
                "available_iterations": item.available_iterations,
                "included_count": item.included_count,
                "included_rate": item.included_rate,
            }
            for item in result.work_inclusion
        ],
        "nearest_background_frequency": [
            {
                "author": item.author,
                "work": item.work_id,
                "nearest_count": item.nearest_count,
                "nearest_rate": item.nearest_rate,
                "included_count": item.included_count,
                "nearest_given_included_rate": item.nearest_given_included_rate,
            }
            for item in result.nearest_background_frequency
        ],
        "limitations": {
            "authenticity_not_proven": True,
            "stability_is_conditional_on_resampling_design": True,
            "empirical_intervals_are_not_formal_confidence_guarantees": True,
            "upstream_feature_selection_not_repeated": True,
            "overlapping_windows_may_not_be_independent": True,
        },
    }


def write_stability_json(
    result: VerificationStabilityResult, *, stream: TextIO
) -> None:
    json.dump(
        stability_json_value(result),
        stream,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    stream.write("\n")


def write_stability_json_path(
    result: VerificationStabilityResult, *, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            write_stability_json(result, stream=stream)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_table(
    path: Path,
    *,
    output_format: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(
                stream, delimiter="," if output_format == "csv" else "\t"
            )
            writer.writerow(columns)
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_replicates(
    result: VerificationStabilityResult, *, path: Path, output_format: str
) -> None:
    base = result.base_result.decision
    rows = tuple(
        (
            item.iteration,
            item.attempt,
            item.iteration_seed,
            item.result.decision.value,
            "true" if item.result.decision is base else "false",
            item.result.query_distance,
            item.result.thresholds.genuine_boundary,
            item.result.thresholds.impostor_boundary,
            item.result.thresholds.accept_threshold,
            item.result.thresholds.reject_threshold,
            item.result.candidate_reference_work_count,
            item.result.background_work_count,
            len(item.selected_feature_names),
            len(item.result.retained_feature_names),
            len(item.result.dropped_zero_variance_features),
            item.result.nearest_background.author,
            item.result.nearest_background.work_id,
            item.result.nearest_background.distance,
            item.result.nearest_background.candidate_vs_background_margin,
            "|".join(item.candidate_reference_works),
            "|".join(item.background_works),
            item.selected_features_sha256,
            item.retained_features_sha256,
        )
        for item in result.replicates
    )
    _atomic_table(
        path, output_format=output_format, columns=REPLICATE_COLUMNS, rows=rows
    )


def write_feature_stability(
    result: VerificationStabilityResult, *, path: Path, output_format: str
) -> None:
    rows = tuple(
        (
            item.feature,
            item.selected_count,
            item.selected_rate,
            item.retained_count,
            item.retained_rate,
            item.retained_given_selected_rate,
        )
        for item in result.feature_stability
    )
    _atomic_table(path, output_format=output_format, columns=FEATURE_COLUMNS, rows=rows)
