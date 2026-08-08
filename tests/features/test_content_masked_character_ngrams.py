from __future__ import annotations

from collections import Counter
from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.features.content_masked_character_ngrams import (
    compute_content_masked_character_ngram_features,
    select_content_masked_character_ngram_vocabulary,
)
from nlpo_toolkit.corpus_analysis.features.models import (
    AnalyzedFeatureCorpus,
    ContentMaskedCharacterNgramOptions,
)


def _corpus(label: str, text: str, upos: str = "NOUN") -> AnalyzedFeatureCorpus:
    record = NLPAnalysisRecord(
        0, 0, 0, 0, 0, len(text), 0, len(text), "", text, text, upos
    )
    source = PreparedCorpus(label, (Path(f"{label}.txt"),), text, text, Counter())
    return AnalyzedFeatureCorpus(source, (record,), (record,), text=text)


def test_selection_is_per_size_frequency_then_lexical_and_columns_are_canonical() -> (
    None
):
    vocabulary = select_content_masked_character_ngram_vocabulary(
        (_corpus("a", "aaaa"), _corpus("b", "bb")),
        options=ContentMaskedCharacterNgramOptions((2, 3), top=2),
    )
    assert tuple((term.size, term.value) for term in vocabulary.terms) == (
        (2, "**"),
        (3, "***"),
    )
    assert vocabulary.terms[0].column_name == "char_masked2__u00002a__u00002a_"
    assert vocabulary.terms[0].column_name.isascii()


def test_relative_frequency_uses_all_positions_and_short_stream_is_zero() -> None:
    vocabulary = select_content_masked_character_ngram_vocabulary(
        (_corpus("fit", "aaaa et"),),
        options=ContentMaskedCharacterNgramOptions((2, 7), top=1),
    )
    values = compute_content_masked_character_ngram_features(
        _corpus("row", "aaaa et"), vocabulary=vocabulary
    )
    first, second = vocabulary.terms
    assert values[first.column_name] == 4.0 / 6.0
    assert values[second.column_name] == 1.0
    short_values = compute_content_masked_character_ngram_features(
        _corpus("short", "a"), vocabulary=vocabulary
    )
    assert short_values[second.column_name] == 0.0
    assert all(isinstance(value, float) for value in values.values())
