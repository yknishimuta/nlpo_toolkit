from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from ..analysis_records import NLPAnalysisRecord
from .filtering import feature_field_value, safe_feature_name
from .models import AnalyzedFeatureCorpus, FeatureField, FeatureScalar


def select_mfw_terms(
    corpora: Sequence[AnalyzedFeatureCorpus],
    *,
    count: int,
    field: FeatureField,
) -> tuple[str, ...]:
    if count <= 0:
        return ()
    frequencies: Counter[str] = Counter()
    for corpus in corpora:
        frequencies.update(
            value
            for record in corpus.lexical_records
            if (value := feature_field_value(record, field))
        )
    return tuple(
        term
        for term, _frequency in sorted(
            frequencies.items(), key=lambda item: (-item[1], item[0])
        )[:count]
    )


def compute_mfw_features(
    records: Sequence[NLPAnalysisRecord],
    *,
    terms: Sequence[str],
    field: FeatureField,
) -> Mapping[str, FeatureScalar]:
    denominator = len(records)
    frequencies = Counter(
        value
        for record in records
        if (value := feature_field_value(record, field))
    )
    return {
        f"mfw_{safe_feature_name(term)}": frequencies.get(term, 0) / denominator
        if denominator
        else 0.0
        for term in terms
    }
