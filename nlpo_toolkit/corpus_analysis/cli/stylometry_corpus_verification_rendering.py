from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TextIO

from ..features.corpus_verification_models import CorpusVerificationResult
from .stylometry_verification_rendering import verification_json_value


def corpus_verification_json_value(result: CorpusVerificationResult) -> dict[str, object]:
    value = verification_json_value(result.verification)
    value["input_mode"] = "corpus"
    value["feature_vocabulary"] = {
        "fit_scope": result.vocabulary.fit_scope,
        "query_excluded": True,
        "query_work": result.vocabulary.query_work,
        "selected_feature_count": result.selected_feature_count,
        "selected_mfw_count": result.vocabulary.selected_mfw_count,
        "selected_character_ngram_count": result.vocabulary.selected_character_ngram_count,
        "selected_upos_ngram_count": result.vocabulary.selected_upos_ngram_count,
        "selected_morphology_count": result.vocabulary.selected_morphology_count,
        "sha256": result.vocabulary.sha256,
    }
    limitations = value["limitations"]
    if not isinstance(limitations, dict):
        raise TypeError("verification limitations must be an object")
    limitations["query_excluded_from_vocabulary_fit"] = True
    return value


def write_corpus_verification_json(
    result: CorpusVerificationResult, *, stream: TextIO
) -> None:
    json.dump(
        corpus_verification_json_value(result),
        stream,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    stream.write("\n")


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


def write_corpus_verification_vocabulary_audit(
    result: CorpusVerificationResult, *, path: Path
) -> None:
    audit = result.vocabulary
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "query_work": audit.query_work,
            "fit_scope": audit.fit_scope,
            "mfw_terms": list(audit.mfw_terms),
            "character_ngrams": [
                {
                    "mode": term.mode.value,
                    "size": term.size,
                    "value": term.value,
                    "column_name": term.column_name,
                }
                for term in audit.character_ngrams
            ],
            "upos_ngrams": [
                {
                    "size": term.size,
                    "values": list(term.tags),
                    "column_name": term.column_name,
                }
                for term in audit.upos_ngrams
            ],
            "morphology": (
                {
                    "attributes": list(audit.morphology.attributes),
                    "values": [
                        {"attribute": item.attribute, "value": item.value}
                        for item in audit.morphology.values
                    ],
                    "bundles": [
                        [
                            {"attribute": item.attribute, "value": item.value}
                            for item in bundle.features
                        ]
                        for bundle in audit.morphology.bundles
                    ],
                }
                if audit.morphology is not None
                else None
            ),
            "selected_mfw_count": audit.selected_mfw_count,
            "selected_character_ngram_count": audit.selected_character_ngram_count,
            "selected_upos_ngram_count": audit.selected_upos_ngram_count,
            "selected_morphology_count": audit.selected_morphology_count,
            "sha256": audit.sha256,
        },
    )
