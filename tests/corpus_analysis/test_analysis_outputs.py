from __future__ import annotations

from collections import Counter

from nlpo_toolkit.corpus_analysis.analysis_outputs import (
    apply_lemma_normalization,
    split_known_unknown,
)


def test_counter_transforms_preserve_existing_behavior() -> None:
    source = Counter({"omninus": 2, "omnino": 1})
    normalized = apply_lemma_normalization(source, {"omninus": "omnino"})
    split = split_known_unknown(Counter({"arma": 2, "ignotus": 1}), {"arma"})

    assert normalized == Counter({"omnino": 3})
    assert source == Counter({"omninus": 2, "omnino": 1})
    assert split.known == Counter({"arma": 2})
    assert split.unknown == Counter({"ignotus": 1})
