from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.features import (
    FeatureFilterPolicy,
    FeatureOptions,
    build_feature_rows,
    compute_basic_features,
    compute_mfw_features,
    compute_upos_features,
    filter_feature_records,
    select_mfw,
)
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken


def _record(
    token: str,
    lemma: str | None = None,
    upos: str | None = "NOUN",
    index: int = 0,
) -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        chunk_index=0,
        sentence_index=0,
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
    row = compute_basic_features(
        records,
        ".",
        "g",
        2,
        raw_token_count=7,
        sentence_count=2,
    )
    assert row["file_count"] == 2
    assert row["token_count"] == 7
    assert row["word_token_count"] == 1
    assert row["mean_sentence_length"] == 0.5
    assert row["type_token_ratio"] == 1.0


def test_empty_filtered_records_are_safe_for_all_calculators() -> None:
    basic = compute_basic_features(
        (),
        "xiv",
        "g",
        1,
        raw_token_count=1,
        sentence_count=1,
    )
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
    assert select_mfw([()], 10, "lemma") == []
    assert compute_mfw_features((), ["rosa"], "lemma") == {"mfw_rosa": 0.0}


def test_upos_and_mfw_denominators_use_the_same_filtered_records() -> None:
    records = (
        _record("rosa", None, "NOUN", index=0),
        _record("amat", "amo", None, index=1),
    )
    upos = compute_upos_features(records)
    assert upos["upos_NOUN_count"] == 1
    assert upos["upos_NOUN_ratio"] == 0.5
    assert upos["content_word_ratio"] == 0.5
    assert select_mfw([records], 2, "lemma") == ["amo", "rosa"]
    assert compute_mfw_features(records, ["amo", "rosa"], "lemma") == {
        "mfw_amo": 0.5,
        "mfw_rosa": 0.5,
    }


def test_mfw_tie_breaking_remains_deterministic() -> None:
    records = (_record("beta", "beta"), _record("alpha", "alpha"))
    assert select_mfw([records], 2, "lemma") == ["alpha", "beta"]


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
    source = [("g", [Path("a.txt")], "xiv a Rosa .")]
    lemma_row = build_feature_rows(
        source,
        NLP(),
        FeatureOptions(field="lemma", mfw=1, filter_policy=policy),
    )[0]
    token_row = build_feature_rows(
        source,
        NLP(),
        FeatureOptions(field="token", mfw=1, filter_policy=policy),
    )[0]

    lemma_shared = {key: value for key, value in lemma_row.items() if not key.startswith("mfw_")}
    token_shared = {key: value for key, value in token_row.items() if not key.startswith("mfw_")}
    assert lemma_shared == token_shared
    assert lemma_row["word_token_count"] == 1
    assert lemma_row["upos_NOUN_count"] == 1
    assert lemma_row["mfw_rose"] == 1.0
    assert token_row["mfw_rosa"] == 1.0
