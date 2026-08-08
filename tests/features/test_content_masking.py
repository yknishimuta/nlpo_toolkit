from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.features.content_masking import (
    build_content_masked_character_stream,
    build_content_masked_text,
    should_mask_content_record,
)
from nlpo_toolkit.corpus_analysis.features.errors import FeatureError
from nlpo_toolkit.corpus_analysis.features.models import (
    AnalyzedFeatureCorpus,
    ContentMaskedCharacterNgramOptions,
    FeatureSamplingOptions,
)
from nlpo_toolkit.corpus_analysis.features.sampling import sample_feature_corpus


def _record(token: str, upos: str | None, start: int, end: int, index: int = 0):
    return NLPAnalysisRecord(
        0, 0, index, index, start, end, start, end, "", token, token, upos
    )


def _corpus(text: str, records: tuple[NLPAnalysisRecord, ...], lexical=None):
    source = PreparedCorpus("work", (Path("work.txt"),), text, text, Counter())
    return AnalyzedFeatureCorpus(source, records, lexical or records, text=text)


@pytest.mark.parametrize("sizes", ((3,), (3, 4, 5)))
def test_content_masked_options_accept_and_preserve_sizes(sizes) -> None:
    options = ContentMaskedCharacterNgramOptions(list(sizes))
    assert options.sizes == sizes
    assert options.top == 300
    with pytest.raises(FrozenInstanceError):
        options.top = 1


@pytest.mark.parametrize("sizes", ((0,), (-1,), (True,), (3, 3)))
def test_content_masked_options_reject_invalid_sizes(sizes) -> None:
    with pytest.raises(FeatureError):
        ContentMaskedCharacterNgramOptions(sizes)


@pytest.mark.parametrize("top", (0, -1, True))
def test_content_masked_options_reject_invalid_top(top) -> None:
    with pytest.raises(FeatureError):
        ContentMaskedCharacterNgramOptions((3,), top)


@pytest.mark.parametrize("upos", ("NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "X"))
def test_content_upos_is_masked(upos: str) -> None:
    assert should_mask_content_record(_record("Rosa", upos, 0, 4))
    assert should_mask_content_record(_record("Rosa", f" {upos.lower()} ", 0, 4))


@pytest.mark.parametrize(
    "upos", ("AUX", "PRON", "ADP", "CCONJ", "SCONJ", "PART", "DET", "PUNCT")
)
def test_function_oriented_upos_is_retained(upos: str) -> None:
    assert not should_mask_content_record(_record("et", upos, 0, 2))


def test_missing_upos_masks_words_but_not_punctuation() -> None:
    assert should_mask_content_record(_record("foobar", None, 0, 6))
    assert not should_mask_content_record(_record(",", " ", 0, 1))


def test_build_content_masked_text_exact_example_and_normalization() -> None:
    text = "Rosa ET puella amat."
    records = (
        _record("Rosa", "NOUN", 0, 4, 0),
        _record("ET", "CCONJ", 5, 7, 1),
        _record("puella", "NOUN", 8, 14, 2),
        _record("amat", "VERB", 15, 19, 3),
        _record(".", "PUNCT", 19, 20, 4),
    )
    corpus = _corpus(text, records)
    assert build_content_masked_text(corpus) == "**** ET ****** ****."
    assert build_content_masked_character_stream(corpus) == "**** et ****** ****."
    assert text == "Rosa ET puella amat."
    assert records[0].token == "Rosa"


def test_mask_preserves_internal_punctuation_and_masks_marks_numbers() -> None:
    text = "a\u0304-12"
    corpus = _corpus(text, (_record(text, "NUM", 0, len(text)),))
    assert build_content_masked_text(corpus) == "**-**"


def test_offsets_not_token_search_and_overlap_is_idempotent() -> None:
    text = "rosa et rosa"
    records = (
        _record("not-the-surface", "NOUN", 8, 12, 0),
        _record("overlap", "VERB", 9, 12, 1),
        _record("et", "CCONJ", 5, 7, 2),
    )
    assert build_content_masked_text(_corpus(text, records)) == "rosa et ****"


@pytest.mark.parametrize("start,end", ((None, 4), (0, None), (-1, 4), (4, 3), (0, 99)))
def test_invalid_masked_record_offsets_fail(start, end) -> None:
    record = _record("rosa", "NOUN", 0, 4)
    record = replace(record, char_start_in_text=start, char_end_in_text=end)
    with pytest.raises(FeatureError, match="exact token offsets"):
        build_content_masked_text(_corpus("rosa", (record,)))


def test_fixed_window_masks_only_exact_sample_with_relative_offsets() -> None:
    text = "Rosa, et puella amat."
    raw = (
        _record("Rosa", "NOUN", 0, 4, 0),
        _record(",", "PUNCT", 4, 5, 1),
        _record("et", "CCONJ", 6, 8, 2),
        _record("puella", "NOUN", 9, 15, 3),
        _record("amat", "VERB", 16, 20, 4),
        _record(".", "PUNCT", 20, 21, 5),
    )
    corpus = _corpus(text, raw, (raw[0], raw[2], raw[3], raw[4]))
    samples = sample_feature_corpus(corpus, options=FeatureSamplingOptions(2))
    assert samples[1].text == "puella amat"
    assert build_content_masked_text(samples[1]) == "****** ****"


def test_no_usable_upos_in_full_corpus_fails() -> None:
    corpus = _corpus("foobar", (_record("foobar", None, 0, 6),))
    with pytest.raises(FeatureError, match="require UPOS annotations"):
        build_content_masked_text(corpus)
