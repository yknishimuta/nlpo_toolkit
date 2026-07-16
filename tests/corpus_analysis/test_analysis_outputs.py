from __future__ import annotations

from collections import Counter

from nlpo_toolkit.corpus_analysis.postprocessing.lemma_normalization import apply_lemma_normalization
from nlpo_toolkit.corpus_analysis.postprocessing.dictionary import classify_dictionary_entries


def test_counter_transforms_preserve_existing_behavior() -> None:
    source = Counter({"omninus": 2, "omnino": 1})
    normalized = apply_lemma_normalization(source, {"omninus": "omnino"})
    split = classify_dictionary_entries(Counter({"arma": 2, "ignotus": 1}), {"arma"})

    assert normalized == Counter({"omnino": 3})
    assert source == Counter({"omninus": 2, "omnino": 1})
    assert split.known == Counter({"arma": 2})
    assert split.unknown == Counter({"ignotus": 1})
