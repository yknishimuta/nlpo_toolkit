from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.features.models import AnalyzedFeatureCorpus, FeatureFilterPolicy, FeatureOptions
from nlpo_toolkit.corpus_analysis.features.engine import (
    analyze_feature_corpus,
    build_feature_matrix,
)
from nlpo_toolkit.corpus_analysis.features.filtering import filter_feature_records
from nlpo_toolkit.corpus_analysis.features.lexical import compute_basic_features
from nlpo_toolkit.corpus_analysis.features.mfw import compute_mfw_features, select_mfw_terms
from nlpo_toolkit.corpus_analysis.features.upos import compute_upos_features
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken
from nlpo_toolkit.corpus_analysis.analysis_policy import AnalysisExtractionPolicy


def _analyzed(
    lexical_records,
    text: str,
    *,
    raw_records=None,
    files: int = 1,
):
    source = PreparedCorpus("g", tuple(Path(f"{index}.txt") for index in range(files)), text, text, Counter())
    lexical = tuple(lexical_records)
    return AnalyzedFeatureCorpus(
        source=source,
        raw_records=tuple(raw_records) if raw_records is not None else lexical,
        lexical_records=lexical,
    )


def _record(
    token: str,
    lemma: str | None = None,
    upos: str | None = "NOUN",
    index: int = 0,
    sentence: int = 0,
    chunk: int = 0,
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


def test_analyzed_corpus_owns_two_immutable_record_populations() -> None:
    first = _record("rosa", index=0, sentence=0, chunk=0)
    punctuation = _record(".", ".", "PUNCT", index=1, sentence=0, chunk=0)
    second = _record("amat", "amo", "VERB", index=2, sentence=0, chunk=1)
    raw = [first, punctuation, second]
    lexical = [first, second]
    analyzed = AnalyzedFeatureCorpus(
        source=PreparedCorpus("g", (Path("a.txt"),), "", "", Counter()),
        raw_records=raw,
        lexical_records=lexical,
    )

    assert analyzed.raw_records == (first, punctuation, second)
    assert analyzed.lexical_records == (first, second)
    assert analyzed.raw_record_count == 3
    assert analyzed.lexical_record_count == 2
    assert analyzed.sentence_count == 2
    assert isinstance(analyzed.raw_records, tuple)
    assert isinstance(analyzed.lexical_records, tuple)
    raw.clear()
    lexical.clear()
    assert analyzed.raw_record_count == 3
    with pytest.raises(FrozenInstanceError):
        analyzed.raw_records = ()  # type: ignore[misc]


def test_analyzed_corpus_empty_raw_population_has_no_sentences() -> None:
    analyzed = _analyzed((), "")
    assert analyzed.raw_record_count == 0
    assert analyzed.lexical_record_count == 0
    assert analyzed.sentence_count == 0


def test_filter_preserves_order_and_returns_tuple_without_mutating_input() -> None:
    records = [
        _record("Rosa", "rosa", index=0),
        _record(".", ".", "PUNCT", index=1),
        _record("amat", "amo", "VERB", index=2),
    ]
    original = list(records)

    filtered = filter_feature_records(records, policy=FeatureFilterPolicy())

    assert isinstance(filtered, tuple)
    assert filtered == (records[0], records[2])
    assert records == original


@pytest.mark.parametrize("token", ["", "   ", ".", "—", "..."])
def test_filter_removes_empty_and_punctuation_only_tokens(token: str) -> None:
    assert filter_feature_records(
        [_record(token, token)], policy=FeatureFilterPolicy()
    ) == ()


def test_filter_keeps_tokens_containing_alphanumeric_characters() -> None:
    record = _record("A1", "a1")
    assert filter_feature_records([record], policy=FeatureFilterPolicy()) == (record,)


def test_minimum_length_is_surface_based_and_inclusive_at_boundary() -> None:
    short = _record("a", "verylong", index=0)
    boundary = _record("ab", "x", index=1)
    assert filter_feature_records(
        [short, boundary],
        policy=FeatureFilterPolicy(min_token_length=2),
    ) == (boundary,)


def test_roman_filter_disabled_keeps_numeral_and_enabled_drops_it() -> None:
    roman = _record("XIV", "word")
    assert filter_feature_records(
        [roman], policy=FeatureFilterPolicy(drop_roman_numerals=False)
    ) == (roman,)
    assert filter_feature_records(
        [roman], policy=FeatureFilterPolicy(drop_roman_numerals=True)
    ) == ()


def test_configured_and_shared_roman_exceptions_are_case_insensitive() -> None:
    xiv = _record("XIV", "xiv", index=0)
    vi = _record("VI", "vi", index=1)
    policy = FeatureFilterPolicy(
        drop_roman_numerals=True,
        roman_exceptions=frozenset({"  XiV  "}),
    )
    assert policy.roman_exceptions == frozenset({"xiv"})
    assert filter_feature_records([xiv, vi], policy=policy) == (xiv, vi)


def test_filter_eligibility_uses_surface_regardless_of_lemma() -> None:
    roman_surface = _record("xiv", "rosa", index=0)
    lexical_surface = _record("rosa", "xiv", index=1)
    assert filter_feature_records(
        [roman_surface, lexical_surface],
        policy=FeatureFilterPolicy(drop_roman_numerals=True),
    ) == (lexical_surface,)


def test_analysis_retains_raw_records_and_filters_one_lexical_population() -> None:
    calls: list[str] = []

    class NLP:
        def __call__(self, text: str) -> NLPDocument:
            calls.append(text)
            return NLPDocument(
                sentences=(
                    NLPSentence(
                        text=text,
                        tokens=(
                            NLPToken(".", ".", "PUNCT"),
                            NLPToken("a", "a", "NOUN"),
                            NLPToken("XV", "xv", "NUM"),
                            NLPToken("XIV", "xiv", "NUM"),
                            NLPToken("rosa", "rosa", None),
                        ),
                    ),
                ),
                text=text,
            )

    source = PreparedCorpus(
        "g", (Path("a.txt"),), "ignored", "ignored", Counter()
    )
    analyzed = analyze_feature_corpus(
        source,
        nlp=NLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        filter_policy=FeatureFilterPolicy(
            min_token_length=2,
            drop_roman_numerals=True,
            roman_exceptions=frozenset({"xiv"}),
        ),
    )

    assert calls == ["ignored"]
    assert [record.token for record in analyzed.raw_records] == [
        ".",
        "a",
        "XV",
        "XIV",
        "rosa",
    ]
    assert [record.token for record in analyzed.lexical_records] == ["XIV", "rosa"]


def test_filter_policy_validates_and_is_frozen() -> None:
    with pytest.raises(ValueError):
        FeatureFilterPolicy(min_token_length=-1)
    with pytest.raises(TypeError):
        FeatureFilterPolicy(min_token_length=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        FeatureFilterPolicy(drop_roman_numerals=1)  # type: ignore[arg-type]
    policy = FeatureFilterPolicy()
    with pytest.raises(FrozenInstanceError):
        policy.min_token_length = 1  # type: ignore[misc]


def test_basic_calculator_trusts_filtered_contract_and_structural_metadata() -> None:
    records = (_record(".", ".", "PUNCT"),)
    raw_records = records + tuple(
        _record(f"raw{index}", sentence=index % 2, index=index + 1)
        for index in range(6)
    )
    row = compute_basic_features(
        _analyzed(records, ".", raw_records=raw_records, files=2)
    )
    assert row["file_count"] == 2
    assert row["token_count"] == 7
    assert row["word_token_count"] == 1
    assert row["mean_sentence_length"] == 0.5
    assert row["type_token_ratio"] == 1.0


def test_empty_filtered_records_are_safe_for_all_calculators() -> None:
    analyzed = _analyzed((), "xiv", raw_records=(_record("xiv"),))
    basic = compute_basic_features(analyzed)
    upos = compute_upos_features(())
    assert basic["token_count"] == 1
    assert basic["word_token_count"] == 0
    for key in (
        "hapax_lemma_ratio",
        "mean_sentence_length",
        "mean_token_length",
        "type_token_ratio",
        "lemma_type_token_ratio",
    ):
        assert basic[key] == 0.0
    assert all(value == 0 or value == 0.0 for value in upos.values())
    assert select_mfw_terms([analyzed], count=10, field="lemma") == ()
    assert compute_mfw_features((), terms=["rosa"], field="lemma") == {"mfw_rosa": 0.0}


def test_upos_and_mfw_denominators_use_the_same_filtered_records() -> None:
    records = (
        _record("rosa", None, "NOUN", index=0),
        _record("amat", "amo", None, index=1),
    )
    upos = compute_upos_features(records)
    assert upos["upos_NOUN_count"] == 1
    assert upos["upos_NOUN_ratio"] == 0.5
    assert upos["content_word_ratio"] == 0.5
    analyzed = _analyzed(records, "rosa amat")
    assert select_mfw_terms([analyzed], count=2, field="lemma") == ("amo", "rosa")
    assert compute_mfw_features(records, terms=["amo", "rosa"], field="lemma") == {
        "mfw_amo": 0.5,
        "mfw_rosa": 0.5,
    }


def test_mfw_tie_breaking_remains_deterministic() -> None:
    records = (_record("beta", "beta"), _record("alpha", "alpha"))
    analyzed = _analyzed(records, "beta alpha")
    assert select_mfw_terms([analyzed], count=2, field="lemma") == ("alpha", "beta")


def test_mfw_field_changes_values_but_not_basic_or_upos_population() -> None:
    class NLP:
        def __call__(self, text: str) -> NLPDocument:
            return NLPDocument(
                sentences=[
                    NLPSentence(
                        text=text,
                        tokens=[
                            NLPToken("xiv", "number", "NUM"),
                            NLPToken("a", "long-lemma", "NOUN"),
                            NLPToken("Rosa", "rose", "NOUN"),
                            NLPToken(".", ".", "PUNCT"),
                        ],
                    )
                ],
                text=text,
            )

    policy = FeatureFilterPolicy(
        min_token_length=2,
        drop_roman_numerals=True,
    )
    source = (
        PreparedCorpus("g", (Path("a.txt"),), "xiv a Rosa .", "xiv a Rosa .", Counter()),
    )
    lemma_row = build_feature_matrix(
        corpora=source,
        nlp=NLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(field="lemma", mfw=1, filter_policy=policy),
    )[0]
    token_row = build_feature_matrix(
        corpora=source,
        nlp=NLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(field="token", mfw=1, filter_policy=policy),
    )[0]

    lemma_shared = {key: value for key, value in lemma_row.items() if not key.startswith("mfw_")}
    token_shared = {key: value for key, value in token_row.items() if not key.startswith("mfw_")}
    assert lemma_shared == token_shared
    assert lemma_row["word_token_count"] == 1
    assert lemma_row["upos_NOUN_count"] == 1
    assert lemma_row["mfw_rose"] == 1.0
    assert token_row["mfw_rosa"] == 1.0
