from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.features.lexical import (
    build_sentence_lengths,
    compute_basic_features,
)
from nlpo_toolkit.corpus_analysis.features.models import AnalyzedFeatureCorpus


def _record(
    token: str,
    *,
    sentence: int,
    chunk: int = 0,
    lemma: str | None = None,
    upos: str | None = "NOUN",
    index: int = 0,
) -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        chunk_index=chunk,
        sentence_index=sentence,
        token_index=index,
        global_token_index=index,
        char_start_in_chunk=None,
        char_end_in_chunk=None,
        char_start_in_text=None,
        char_end_in_text=None,
        sentence="",
        token=token,
        lemma=lemma,
        upos=upos,
    )


def _analyzed(
    raw_records: tuple[NLPAnalysisRecord, ...],
    lexical_records: tuple[NLPAnalysisRecord, ...],
) -> AnalyzedFeatureCorpus:
    return AnalyzedFeatureCorpus(
        source=PreparedCorpus("g", (Path("a.txt"),), "prepared", "prepared", Counter()),
        raw_records=raw_records,
        lexical_records=lexical_records,
    )


def test_sentence_length_distribution_includes_zero_length_raw_sentences() -> None:
    lexical = tuple(
        _record(f"w{sentence}{index}", sentence=sentence, index=index)
        for sentence, length in enumerate((1, 2, 3, 4))
        for index in range(length)
    )
    punctuation_only = _record(".", sentence=4, upos="PUNCT", index=10)
    raw = lexical + (punctuation_only,)

    assert build_sentence_lengths(raw_records=raw, lexical_records=lexical) == (
        1,
        2,
        3,
        4,
        0,
    )

    row = compute_basic_features(_analyzed(raw, lexical))
    assert row["word_token_count"] == 10
    assert row["mean_sentence_length"] == 2.0
    assert row["sentence_length_variance"] == pytest.approx(2.0)
    assert row["sentence_length_median"] == 2.0
    assert row["sentence_length_q25"] == 1.0
    assert row["sentence_length_q75"] == 3.0


def test_sentence_distribution_uses_chunk_and_sentence_index() -> None:
    first = _record("a", sentence=0, chunk=0)
    second = _record("bb", sentence=0, chunk=1)

    assert build_sentence_lengths(
        raw_records=(first, second), lexical_records=(first, second)
    ) == (1, 1)


def test_sentence_distribution_for_four_lengths_matches_specification() -> None:
    lexical = tuple(
        _record("x", sentence=sentence, index=index)
        for sentence, length in enumerate((1, 2, 3, 4))
        for index in range(length)
    )
    row = compute_basic_features(_analyzed(lexical, lexical))

    assert row["mean_sentence_length"] == 2.5
    assert row["sentence_length_variance"] == 1.25
    assert row["sentence_length_median"] == 2.5
    assert row["sentence_length_q25"] == 1.75
    assert row["sentence_length_q75"] == 3.25


def test_token_length_distribution_uses_stripped_surface_not_lemma() -> None:
    lexical = (
        _record(" a ", sentence=0, lemma="long-lemma"),
        _record("BB", sentence=0, lemma=None),
        _record("ccc", sentence=0, lemma="x"),
        _record("DDDD", sentence=0, lemma="y"),
    )
    punctuation = _record("........", sentence=0, upos="PUNCT")
    row = compute_basic_features(_analyzed(lexical + (punctuation,), lexical))

    assert row["mean_token_length"] == 2.5
    assert row["token_length_variance"] == 1.25
    assert row["token_length_median"] == 2.5
    assert row["token_length_q25"] == 1.75
    assert row["token_length_q75"] == 3.25


def test_empty_and_singleton_length_distributions_are_total() -> None:
    empty = compute_basic_features(_analyzed((), ()))
    singleton_record = _record("arma", sentence=0)
    singleton = compute_basic_features(
        _analyzed((singleton_record,), (singleton_record,))
    )

    for prefix in ("sentence_length", "token_length"):
        assert empty[f"{prefix}_variance"] == 0.0
        assert empty[f"{prefix}_median"] == 0.0
        assert empty[f"{prefix}_q25"] == 0.0
        assert empty[f"{prefix}_q75"] == 0.0
    assert singleton["sentence_length_variance"] == 0.0
    assert singleton["sentence_length_median"] == 1.0
    assert singleton["sentence_length_q25"] == 1.0
    assert singleton["sentence_length_q75"] == 1.0
    assert singleton["token_length_variance"] == 0.0
    assert singleton["token_length_median"] == 4.0
    assert singleton["token_length_q25"] == 4.0
    assert singleton["token_length_q75"] == 4.0


def test_basic_length_columns_have_the_required_order_and_float_values() -> None:
    record = _record("arma", sentence=0)
    row = compute_basic_features(_analyzed((record,), (record,)))
    expected = (
        "mean_sentence_length",
        "sentence_length_variance",
        "sentence_length_median",
        "sentence_length_q25",
        "sentence_length_q75",
        "mean_token_length",
        "token_length_variance",
        "token_length_median",
        "token_length_q25",
        "token_length_q75",
    )
    keys = tuple(row)

    assert tuple(key for key in keys if key in expected) == expected
    assert all(isinstance(row[key], float) for key in expected)
