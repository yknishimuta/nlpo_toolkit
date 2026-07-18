from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.features.errors import FeatureError
from nlpo_toolkit.corpus_analysis.features.models import (
    AnalyzedFeatureCorpus,
    UposNgramOptions,
)
from nlpo_toolkit.corpus_analysis.features.upos_ngrams import (
    UposNgramTerm,
    UposNgramVocabulary,
    compute_upos_ngram_features,
    iter_upos_ngrams,
    iter_upos_runs,
    normalize_upos_value,
    select_upos_ngram_vocabulary,
)


def _record(
    upos: str | None,
    sentence: int = 0,
    *,
    chunk: int = 0,
    token: str = "x",
) -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        chunk_index=chunk,
        sentence_index=sentence,
        token_index=0,
        global_token_index=0,
        char_start_in_chunk=None,
        char_end_in_chunk=None,
        char_start_in_text=None,
        char_end_in_text=None,
        sentence="ignored",
        token=token,
        lemma=token,
        upos=upos,
    )


def _corpus(
    label: str, records: tuple[NLPAnalysisRecord, ...]
) -> AnalyzedFeatureCorpus:
    source = PreparedCorpus(label, (Path(f"{label}.txt"),), label, label, Counter())
    return AnalyzedFeatureCorpus(source, records, records, text=label)


def test_options_are_frozen_ordered_and_strict() -> None:
    options = UposNgramOptions([3, 2])
    assert options.sizes == (3, 2)
    assert options.top == 100
    with pytest.raises(FrozenInstanceError):
        options.top = 3  # type: ignore[misc]
    for sizes in ((), (0,), (1,), (4,), (True,), (2, 2)):
        with pytest.raises(FeatureError):
            UposNgramOptions(sizes)
    for top in (0, -1, True):
        with pytest.raises(FeatureError):
            UposNgramOptions((2,), top)


def test_upos_normalization() -> None:
    assert normalize_upos_value(" noun ") == "NOUN"
    assert normalize_upos_value("X") == "X"
    assert normalize_upos_value(" custom ") == "CUSTOM"
    assert normalize_upos_value("") is None
    assert normalize_upos_value("  ") is None
    assert normalize_upos_value(None) is None


def test_runs_split_on_sentence_chunk_and_missing_upos() -> None:
    records = (
        _record("noun", 0),
        _record("VERB", 0),
        _record("ADP", 1),
        _record(None, 1),
        _record("NOUN", 1),
        _record("X", 0, chunk=1),
    )
    assert tuple(iter_upos_runs(records)) == (
        ("NOUN", "VERB"),
        ("ADP",),
        ("NOUN",),
        ("X",),
    )
    assert records[0].sentence == "ignored"


def test_reappearing_sentence_id_is_rejected() -> None:
    records = (_record("NOUN", 0), _record("VERB", 1), _record("ADJ", 0))
    with pytest.raises(FeatureError, match="reappeared"):
        tuple(iter_upos_runs(records))


def test_upos_ngram_iteration_boundaries() -> None:
    run = ("ADP", "DET", "NOUN")
    assert tuple(iter_upos_ngrams(run, size=2)) == (
        ("ADP", "DET"),
        ("DET", "NOUN"),
    )
    assert tuple(iter_upos_ngrams(run, size=3)) == (("ADP", "DET", "NOUN"),)
    assert tuple(iter_upos_ngrams(("NOUN",), size=2)) == ()
    assert run == ("ADP", "DET", "NOUN")


def test_vocabulary_pools_unsampled_corpora_and_tie_breaks() -> None:
    first = _corpus("a", (_record("NOUN"), _record("VERB"), _record("NOUN")))
    second = _corpus("b", (_record("NOUN"), _record("VERB"), _record("ADJ")))
    vocabulary = select_upos_ngram_vocabulary(
        (first, second), options=UposNgramOptions((2, 3), top=3)
    )
    assert [(term.size, term.tags) for term in vocabulary.terms] == [
        (2, ("NOUN", "VERB")),
        (2, ("VERB", "ADJ")),
        (2, ("VERB", "NOUN")),
        (3, ("NOUN", "VERB", "ADJ")),
        (3, ("NOUN", "VERB", "NOUN")),
    ]
    assert vocabulary.terms[0].column_name == "upos2_NOUN_VERB"
    assert vocabulary.terms[3].column_name == "upos3_NOUN_VERB_ADJ"
    with pytest.raises(FeatureError, match="no UPOS 3-grams"):
        select_upos_ngram_vocabulary(
            (_corpus("short", (_record("NOUN"), _record("VERB", 1))),),
            options=UposNgramOptions((3,)),
        )


def test_relative_frequency_uses_all_positions_of_each_size() -> None:
    vocabulary = UposNgramVocabulary(
        (
            UposNgramTerm(2, ("NOUN", "VERB"), "upos2_NOUN_VERB"),
            UposNgramTerm(2, ("VERB", "NOUN"), "upos2_VERB_NOUN"),
            UposNgramTerm(2, ("ADP", "NOUN"), "upos2_ADP_NOUN"),
            UposNgramTerm(3, ("NOUN", "VERB", "NOUN"), "upos3_NOUN_VERB_NOUN"),
        )
    )
    records = (
        _record("NOUN", 0),
        _record("VERB", 0),
        _record("NOUN", 0),
        _record("ADP", 1),
        _record("NOUN", 1),
    )
    result = compute_upos_ngram_features(records, vocabulary=vocabulary)
    assert result["upos2_NOUN_VERB"] == pytest.approx(1 / 3)
    assert result["upos2_VERB_NOUN"] == pytest.approx(1 / 3)
    assert result["upos2_ADP_NOUN"] == pytest.approx(1 / 3)
    assert result["upos3_NOUN_VERB_NOUN"] == 1.0
    assert tuple(result) == tuple(term.column_name for term in vocabulary.terms)


def test_missing_upos_breaks_sequence_and_zero_positions_return_zero() -> None:
    vocabulary = UposNgramVocabulary(
        (UposNgramTerm(2, ("NOUN", "VERB"), "upos2_NOUN_VERB"),)
    )
    result = compute_upos_ngram_features(
        (_record("NOUN"), _record(None), _record("VERB")),
        vocabulary=vocabulary,
    )
    assert result["upos2_NOUN_VERB"] == 0.0
