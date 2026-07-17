from __future__ import annotations

from collections import Counter
from collections.abc import Mapping


def apply_lemma_normalization(
    counter: Mapping[str, int],
    normalization_map: Mapping[str, str] | None,
) -> Counter[str]:
    if normalization_map is None:
        return counter.copy()
    normalized: Counter[str] = Counter()
    for lemma, count in counter.items():
        normalized[normalization_map.get(lemma, lemma)] += count
    return normalized
