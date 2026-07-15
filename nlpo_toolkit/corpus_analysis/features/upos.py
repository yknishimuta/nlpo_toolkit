from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from ..analysis_records import NLPAnalysisRecord
from .models import FeatureScalar


UPOS_FEATURES = ("NOUN", "VERB", "ADJ", "ADV", "PROPN", "PRON", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "DET", "NUM")
CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
FUNCTION_UPOS = {"PRON", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "DET"}


def compute_upos_features(
    records: Sequence[NLPAnalysisRecord],
) -> Mapping[str, FeatureScalar]:
    denominator = len(records)
    counts = Counter(record.upos or "X" for record in records)
    features: dict[str, FeatureScalar] = {}
    for upos in UPOS_FEATURES:
        count = counts.get(upos, 0)
        features[f"upos_{upos}_count"] = count
        features[f"upos_{upos}_ratio"] = count / denominator if denominator else 0.0
    content_count = sum(counts.get(upos, 0) for upos in CONTENT_UPOS)
    function_count = sum(counts.get(upos, 0) for upos in FUNCTION_UPOS)
    features["content_word_count"] = content_count
    features["content_word_ratio"] = content_count / denominator if denominator else 0.0
    features["function_word_count"] = function_count
    features["function_word_ratio"] = function_count / denominator if denominator else 0.0
    return features
