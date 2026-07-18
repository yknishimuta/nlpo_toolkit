from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import TextIO

from ..features.corpus_lowo_models import CorpusLowoResult


FOLD_COLUMNS = (
    "fold_index",
    "test_author",
    "test_work",
    "predicted_author",
    "is_correct",
    "best_distance",
    "runner_up_author",
    "runner_up_distance",
    "margin",
    "training_work_count",
    "selected_feature_count",
    "selected_mfw_count",
    "selected_character_ngram_count",
    "selected_upos_ngram_count",
    "dropped_zero_variance_count",
    "vocabulary_sha256",
)


def write_corpus_lowo_folds(
    result: CorpusLowoResult, *, stream: TextIO, output_format: str
) -> None:
    writer = csv.writer(stream, delimiter="," if output_format == "csv" else "\t")
    writer.writerow(FOLD_COLUMNS)
    for item in result.folds:
        fold = item.evaluation
        writer.writerow(
            (
                fold.fold_index,
                fold.actual_author,
                fold.work_id,
                fold.predicted_author,
                "true" if fold.is_correct else "false",
                fold.best_distance,
                fold.runner_up_author,
                fold.runner_up_distance,
                fold.margin,
                fold.training_work_count,
                item.selected_feature_count,
                item.selected_mfw_count,
                item.selected_character_ngram_count,
                item.selected_upos_ngram_count,
                fold.dropped_feature_count,
                item.vocabulary.sha256,
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


def write_vocabulary_audit(result: CorpusLowoResult, *, path: Path) -> None:
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "folds": [
                {
                    "fold_index": item.vocabulary.fold_index,
                    "test_work": item.vocabulary.test_work,
                    "mfw_terms": list(item.vocabulary.mfw_terms),
                    "character_ngrams": [
                        {
                            "size": term.size,
                            "value": term.value,
                            "column_name": term.column_name,
                        }
                        for term in item.vocabulary.character_ngrams
                    ],
                    "upos_ngrams": [
                        {
                            "size": term.size,
                            "values": list(term.tags),
                            "column_name": term.column_name,
                        }
                        for term in item.vocabulary.upos_ngrams
                    ],
                    "morphology": (
                        {
                            "attributes": list(item.vocabulary.morphology.attributes),
                            "values": [
                                {
                                    "attribute": value.attribute,
                                    "value": value.value,
                                }
                                for value in item.vocabulary.morphology.values
                            ],
                            "bundles": [
                                [
                                    {
                                        "attribute": value.attribute,
                                        "value": value.value,
                                    }
                                    for value in bundle.features
                                ]
                                for bundle in item.vocabulary.morphology.bundles
                            ],
                        }
                        if item.vocabulary.morphology is not None
                        else None
                    ),
                    "sha256": item.vocabulary.sha256,
                }
                for item in result.folds
            ],
        },
    )


def write_corpus_lowo_summary(result: CorpusLowoResult, *, path: Path) -> None:
    summary = result.summary
    _atomic_json(
        path,
        {
            "method": "leave_one_work_out_corpus",
            "classifier": "burrows_delta_author_centroid",
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
        },
    )
