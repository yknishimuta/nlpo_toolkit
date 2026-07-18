from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from ..analysis_records import NLPAnalysisRecord
from .errors import FeatureError
from .filtering import feature_field_value, safe_feature_name
from .models import FeatureScalar, FunctionWordOptions, FunctionWordVocabulary


def build_function_word_columns(
    vocabulary: FunctionWordVocabulary,
) -> tuple[tuple[str, str], ...]:
    columns: list[tuple[str, str]] = []
    owners: dict[str, str] = {}
    for term in vocabulary.terms:
        suffix = safe_feature_name(term)
        if suffix == "empty" and term != "empty":
            raise FeatureError(
                f"function-word feature column has an empty suffix for term {term!r}"
            )
        column = f"fw_{suffix}"
        previous = owners.get(column)
        if previous is not None:
            raise FeatureError(
                "function-word feature column collision: "
                f"{previous!r} and {term!r} both produce {column!r}"
            )
        owners[column] = term
        columns.append((term, column))
    return tuple(columns)


def compute_function_word_features(
    records: Sequence[NLPAnalysisRecord],
    *,
    options: FunctionWordOptions,
) -> Mapping[str, FeatureScalar]:
    columns = build_function_word_columns(options.vocabulary)
    frequencies = Counter(
        feature_field_value(record, options.field) for record in records
    )
    denominator = len(records)
    return {
        column: frequencies.get(term, 0) / denominator if denominator else 0.0
        for term, column in columns
    }
