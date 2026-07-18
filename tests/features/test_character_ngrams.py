from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.features.character_ngrams import (
    CharacterNgramTerm,
    CharacterNgramVocabulary,
    compute_character_ngram_features,
    iter_character_ngrams,
    select_character_ngram_vocabulary,
)
from nlpo_toolkit.corpus_analysis.features.character_text import (
    encode_character_ngram,
    feature_unit_character_text,
    normalize_character_stream,
)
from nlpo_toolkit.corpus_analysis.features.engine import (
    prepare_character_ngram_vocabulary,
)
from nlpo_toolkit.corpus_analysis.features.errors import FeatureError
from nlpo_toolkit.corpus_analysis.features.models import (
    AnalyzedFeatureCorpus,
    CharacterNgramOptions,
    FeatureSamplingOptions,
)
from nlpo_toolkit.corpus_analysis.features.sampling import sample_feature_corpus


def _record(token: str, start: int | None, end: int | None) -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        chunk_index=0,
        sentence_index=0,
        token_index=0,
        global_token_index=0,
        char_start_in_chunk=start,
        char_end_in_chunk=end,
        char_start_in_text=start,
        char_end_in_text=end,
        sentence="",
        token=token,
        lemma=token,
        upos="PUNCT" if token == "," else "NOUN",
    )


def test_character_options_are_frozen_ordered_and_strict() -> None:
    options = CharacterNgramOptions([3, 5], 12)
    assert options.sizes == (3, 5)
    assert options.top == 12
    with pytest.raises(FrozenInstanceError):
        options.top = 2  # type: ignore[misc]
    for sizes in ((), (0,), (-1,), (True,), (3, 3)):
        with pytest.raises(FeatureError):
            CharacterNgramOptions(sizes)
    for top in (0, -1, True):
        with pytest.raises(FeatureError):
            CharacterNgramOptions((3,), top)


def test_normalize_character_stream_preserves_non_whitespace() -> None:
    source = "  Rosa,\n\tAMAT.\u2003X1  "
    assert normalize_character_stream(source) == "rosa, amat. x1"
    assert source == "  Rosa,\n\tAMAT.\u2003X1  "
    assert normalize_character_stream("") == ""


def test_character_ngram_iteration_boundaries() -> None:
    assert tuple(iter_character_ngrams("abcd", size=2)) == ("ab", "bc", "cd")
    assert tuple(iter_character_ngrams("abcd", size=4)) == ("abcd",)
    assert tuple(iter_character_ngrams("ab cd", size=3)) == ("ab ", "b c", " cd")
    assert tuple(iter_character_ngrams("a,b", size=2)) == ("a,", ",b")
    assert tuple(iter_character_ngrams("ab", size=3)) == ()
    assert tuple(iter_character_ngrams("", size=1)) == ()


def test_column_encoder_is_ascii_deterministic_and_unambiguous() -> None:
    assert encode_character_ngram("que") == "que"
    assert encode_character_ngram(" et") == "_sp_et"
    assert encode_character_ngram("a,b") == "a_u00002c_b"
    assert encode_character_ngram("a_b") == "a_u00005f_b"
    assert encode_character_ngram("æst") == "_u0000e6_st"
    values = ("a b", "a-b", "a_b", "a,b", "æst")
    encoded = tuple(encode_character_ngram(value) for value in values)
    assert len(encoded) == len(set(encoded))
    assert all(value.isascii() for value in encoded)


def test_vocabulary_uses_pooled_counts_size_order_and_tie_break() -> None:
    texts = ["aaaa", "aabb"]
    vocabulary = select_character_ngram_vocabulary(
        texts, options=CharacterNgramOptions((2, 3), top=3)
    )
    assert [(term.size, term.value) for term in vocabulary.terms] == [
        (2, "aa"),
        (2, "ab"),
        (2, "bb"),
        (3, "aaa"),
        (3, "aab"),
        (3, "abb"),
    ]
    assert vocabulary.terms[0].column_name == "char2_aa"
    assert texts == ["aaaa", "aabb"]
    with pytest.raises(FeatureError, match="no character 10-grams"):
        select_character_ngram_vocabulary(texts, options=CharacterNgramOptions((10,)))


def test_row_relative_frequencies_use_possible_positions() -> None:
    vocabulary = CharacterNgramVocabulary(
        (
            CharacterNgramTerm(2, "aa", "char2_aa"),
            CharacterNgramTerm(2, "ab", "char2_ab"),
            CharacterNgramTerm(2, "ba", "char2_ba"),
            CharacterNgramTerm(5, "ababa", "char5_ababa"),
        )
    )
    assert (
        compute_character_ngram_features("aaaa", vocabulary=vocabulary)["char2_aa"]
        == 1.0
    )
    result = compute_character_ngram_features("abab", vocabulary=vocabulary)
    assert result["char2_ab"] == pytest.approx(2 / 3)
    assert result["char2_ba"] == pytest.approx(1 / 3)
    assert result["char2_aa"] == 0.0
    assert result["char5_ababa"] == 0.0
    assert tuple(result) == tuple(term.column_name for term in vocabulary.terms)


def test_character_features_require_one_source_file_before_analysis() -> None:
    corpus = PreparedCorpus(
        "g", (Path("a.txt"), Path("b.txt")), "abc", "abc", Counter()
    )
    with pytest.raises(FeatureError, match="group-by-file"):
        prepare_character_ngram_vocabulary(
            (corpus,), options=CharacterNgramOptions((2,))
        )


def test_fixed_sample_uses_exact_prepared_text_span() -> None:
    source = PreparedCorpus("g", (Path("a.txt"),), "ab, cd", "ab, cd", Counter())
    first = _record("ab", 0, 2)
    punctuation = _record(",", 2, 3)
    last = _record("cd", 4, 6)
    analyzed = AnalyzedFeatureCorpus(
        source=source,
        raw_records=(first, punctuation, last),
        lexical_records=(first, last),
        text=source.prepared_text,
    )
    full = sample_feature_corpus(
        analyzed, options=FeatureSamplingOptions(window_tokens=2)
    )
    assert feature_unit_character_text(full[0]) == "ab, cd"
    separate = sample_feature_corpus(
        analyzed, options=FeatureSamplingOptions(window_tokens=1)
    )
    assert [feature_unit_character_text(item) for item in separate] == ["ab", "cd"]


def test_fixed_sample_missing_offsets_has_no_token_join_fallback() -> None:
    source = PreparedCorpus("g", (Path("a.txt"),), "ab cd", "ab cd", Counter())
    first, last = _record("ab", None, None), _record("cd", None, None)
    analyzed = AnalyzedFeatureCorpus(
        source=source,
        raw_records=(first, last),
        lexical_records=(first, last),
        text=source.prepared_text,
    )
    sample = sample_feature_corpus(
        analyzed, options=FeatureSamplingOptions(window_tokens=2)
    )[0]
    with pytest.raises(FeatureError, match="exact text offsets"):
        feature_unit_character_text(sample)
