from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TextIO

from nlpo_toolkit.stylometry.evaluation_results import LeaveOneWorkOutEvaluationResult


FOLD_COLUMNS = (
    "fold_index",
    "work_id",
    "actual_author",
    "predicted_author",
    "correct",
    "test_sample_count",
    "training_work_count",
    "candidate_author_count",
    "retained_feature_count",
    "dropped_zero_variance_count",
    "best_distance",
    "runner_up_author",
    "runner_up_distance",
    "margin",
)


def write_lowo_folds(
    result: LeaveOneWorkOutEvaluationResult,
    *,
    stream: TextIO,
    output_format: str,
) -> None:
    writer = csv.writer(stream, delimiter="," if output_format == "csv" else "\t")
    writer.writerow(FOLD_COLUMNS)
    for fold in result.folds:
        writer.writerow(
            (
                fold.fold_index,
                fold.work_id,
                fold.actual_author,
                fold.predicted_author,
                "true" if fold.is_correct else "false",
                fold.test_sample_count,
                fold.training_work_count,
                fold.candidate_author_count,
                fold.retained_feature_count,
                fold.dropped_feature_count,
                fold.best_distance,
                fold.runner_up_author,
                fold.runner_up_distance,
                fold.margin,
            )
        )


def summary_json_value(result: LeaveOneWorkOutEvaluationResult) -> dict[str, object]:
    summary = result.summary
    return {
        "method": "leave_one_work_out",
        "classifier": "burrows_delta_author_centroid",
        "profile_unit": "work_mean",
        "standardization_fit_unit": "training_works",
        "work_count": summary.work_count,
        "correct_work_count": summary.correct_work_count,
        "accuracy": summary.accuracy,
        "author_count": summary.author_count,
        "macro_author_accuracy": summary.macro_author_accuracy,
        "authors": [
            {
                "author": item.author,
                "work_count": item.work_count,
                "correct_work_count": item.correct_work_count,
                "accuracy": item.accuracy,
            }
            for item in summary.per_author
        ],
    }


def write_lowo_summary_json(
    result: LeaveOneWorkOutEvaluationResult, *, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            summary_json_value(result),
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        stream.write("\n")
