from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from ..analysis_records import NLPAnalysisRecord
from .filtering import feature_lemma_value, feature_token_value
from .models import FeatureScalar, LexicalDiversityOptions


def _ttr(values: Sequence[str]) -> float:
    return len(set(values)) / len(values) if values else 0.0


def compute_mattr(values: Sequence[str], *, window_size: int) -> float:
    count = len(values)
    if count <= window_size:
        return _ttr(values)
    frequencies = Counter(values[:window_size])
    distinct = len(frequencies)
    total = distinct / window_size
    window_count = count - window_size + 1
    for start in range(1, window_count):
        outgoing = values[start - 1]
        frequencies[outgoing] -= 1
        if frequencies[outgoing] == 0:
            del frequencies[outgoing]
            distinct -= 1
        incoming = values[start + window_size - 1]
        if frequencies[incoming] == 0:
            distinct += 1
        frequencies[incoming] += 1
        total += distinct / window_size
    return float(total / window_count)


def compute_msttr(values: Sequence[str], *, segment_size: int) -> float:
    count = len(values)
    if count < segment_size:
        return _ttr(values)
    segment_count = count // segment_size
    total = sum(
        len(set(values[start : start + segment_size])) / segment_size
        for start in range(0, segment_count * segment_size, segment_size)
    )
    return float(total / segment_count)


def _directional_mtld(values: Sequence[str], threshold: float) -> float:
    if not values:
        return 0.0
    frequencies: Counter[str] = Counter()
    factor_tokens = 0
    factors = 0.0
    for value in values:
        frequencies[value] += 1
        factor_tokens += 1
        current_ttr = len(frequencies) / factor_tokens
        if current_ttr <= threshold:
            factors += 1.0
            frequencies.clear()
            factor_tokens = 0
    if factor_tokens:
        remainder_ttr = len(frequencies) / factor_tokens
        factors += (1.0 - remainder_ttr) / (1.0 - threshold)
    if factors == 0.0:
        return float(len(values))
    return float(len(values) / factors)


def compute_mtld(values: Sequence[str], *, threshold: float) -> float:
    if not values:
        return 0.0
    forward = _directional_mtld(values, threshold)
    reverse = _directional_mtld(tuple(reversed(values)), threshold)
    result = (forward + reverse) / 2.0
    return result if math.isfinite(result) and result > 0.0 else float(len(values))


def compute_hdd(values: Sequence[str], *, sample_size: int) -> float:
    population_size = len(values)
    if population_size == 0:
        return 0.0
    effective_size = min(sample_size, population_size)
    probability_sum = 0.0
    for frequency in Counter(values).values():
        if population_size - frequency < effective_size:
            probability_absent = 0.0
        else:
            probability_absent = 1.0
            for index in range(effective_size):
                probability_absent *= (population_size - frequency - index) / (
                    population_size - index
                )
        probability_sum += min(1.0, max(0.0, 1.0 - probability_absent))
    return min(1.0, max(0.0, probability_sum / effective_size))


def compute_lexical_diversity_features(
    records: Sequence[NLPAnalysisRecord],
    *,
    options: LexicalDiversityOptions,
) -> Mapping[str, FeatureScalar]:
    token_values = tuple(feature_token_value(record) for record in records)
    lemma_values = tuple(feature_lemma_value(record) for record in records)
    return {
        "mattr_token": compute_mattr(token_values, window_size=options.window_size),
        "mattr_lemma": compute_mattr(lemma_values, window_size=options.window_size),
        "msttr_token": compute_msttr(token_values, segment_size=options.window_size),
        "msttr_lemma": compute_msttr(lemma_values, segment_size=options.window_size),
        "mtld_token": compute_mtld(token_values, threshold=options.mtld_threshold),
        "mtld_lemma": compute_mtld(lemma_values, threshold=options.mtld_threshold),
        "hdd_token": compute_hdd(token_values, sample_size=options.hdd_sample_size),
        "hdd_lemma": compute_hdd(lemma_values, sample_size=options.hdd_sample_size),
    }
