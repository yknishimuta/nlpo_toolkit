from __future__ import annotations

from collections import Counter
from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_pipeline import (
    apply_lemma_normalization,
    split_known_unknown,
)


def test_analysis_pipeline_counter_helpers_preserve_existing_behavior() -> None:
    normalized = apply_lemma_normalization(
        Counter({"omninus": 2, "omnino": 1}),
        {"omninus": "omnino"},
    )
    known, unknown = split_known_unknown(
        Counter({"arma": 2, "ignotus": 1}),
        {"arma"},
    )

    assert normalized == Counter({"omnino": 3})
    assert known == Counter({"arma": 2})
    assert unknown == Counter({"ignotus": 1})


def test_analysis_pipeline_functions_have_canonical_module() -> None:
    from nlpo_toolkit.corpus_analysis.analysis_pipeline import analyze_corpora

    assert analyze_corpora.__module__.endswith(".analysis_pipeline")
