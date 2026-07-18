from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_policy import AnalysisExtractionPolicy
from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.corpus import PreparedCorpus
from nlpo_toolkit.corpus_analysis.features.engine import build_feature_matrix
from nlpo_toolkit.corpus_analysis.features.errors import FeatureError
from nlpo_toolkit.corpus_analysis.features.models import (
    AnalyzedFeatureCorpus,
    FeatureFilterPolicy,
    FeatureOptions,
    FeatureSamplingOptions,
)
from nlpo_toolkit.corpus_analysis.features.sampling import (
    iter_feature_window_ranges,
    sample_feature_corpus,
)
from nlpo_toolkit.nlp.contracts import NLPDocument, NLPSentence, NLPToken


def _record(
    token: str,
    index: int,
    *,
    upos: str = "NOUN",
    sentence: int = 0,
    start: int | None = None,
    end: int | None = None,
) -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        chunk_index=0,
        sentence_index=sentence,
        token_index=index,
        global_token_index=index,
        char_start_in_chunk=start,
        char_end_in_chunk=end,
        char_start_in_text=start,
        char_end_in_text=end,
        sentence="",
        token=token,
        lemma=token.lower(),
        upos=upos,
    )


def _analyzed(
    raw: tuple[NLPAnalysisRecord, ...],
    eligible_indices: tuple[int, ...],
    *,
    files: tuple[Path, ...] = (Path("a.txt"),),
) -> AnalyzedFeatureCorpus:
    return AnalyzedFeatureCorpus(
        source=PreparedCorpus("g", files, "raw", "prepared", Counter()),
        raw_record_count=len(raw),
        sentence_count=1,
        records=tuple(raw[index] for index in eligible_indices),
        raw_records=raw,
        eligible_raw_indices=eligible_indices,
    )


@pytest.mark.parametrize(
    "kwargs,message",
    (
        ({"window_tokens": 0}, "window_tokens"),
        ({"window_tokens": -1}, "window_tokens"),
        ({"window_tokens": True}, "window_tokens"),
        ({"window_tokens": 2, "step_tokens": 0}, "step_tokens"),
        ({"window_tokens": 2, "step_tokens": True}, "step_tokens"),
        ({"step_tokens": 1}, "requires window_tokens"),
        ({"window_tokens": 2, "include_partial": 1}, "include_partial"),
    ),
)
def test_sampling_options_are_strict(kwargs, message: str) -> None:
    with pytest.raises(FeatureError, match=message):
        FeatureSamplingOptions(**kwargs)


def test_window_ranges_cover_full_overlap_and_one_partial() -> None:
    assert tuple(
        iter_feature_window_ranges(5, options=FeatureSamplingOptions(2))
    ) == ((0, 2, "full"), (2, 4, "full"))
    assert tuple(
        iter_feature_window_ranges(
            5, options=FeatureSamplingOptions(2, include_partial=True)
        )
    ) == ((0, 2, "full"), (2, 4, "full"), (4, 5, "partial"))
    assert tuple(
        iter_feature_window_ranges(5, options=FeatureSamplingOptions(3, 1))
    ) == ((0, 3, "full"), (1, 4, "full"), (2, 5, "full"))
    assert FeatureSamplingOptions(4).effective_step_tokens == 4


def test_sample_uses_raw_span_and_half_open_filtered_offsets() -> None:
    raw = (
        _record("aa", 0, start=0, end=2),
        _record(".", 1, upos="PUNCT", start=2, end=3),
        _record("bb", 2, sentence=1, start=4, end=6),
        _record("cc", 3, sentence=1, start=7, end=9),
    )
    samples = sample_feature_corpus(
        _analyzed(raw, (0, 2, 3)), options=FeatureSamplingOptions(2, 1)
    )
    first, second = samples
    assert [record.token for record in first.records] == ["aa", "bb"]
    assert first.raw_record_count == 3
    assert first.sentence_count == 2
    assert first.char_count == 6
    assert first.sample is not None
    assert (first.sample.sample_index, first.sample.start_token, first.sample.end_token) == (
        1,
        0,
        2,
    )
    assert first.sample.kind == "full"
    assert [record.token for record in second.records] == ["bb", "cc"]


def test_sampling_rejects_multi_file_corpus() -> None:
    raw = (_record("aa", 0), _record("bb", 1))
    with pytest.raises(FeatureError, match="--group-by-file"):
        sample_feature_corpus(
            _analyzed(raw, (0, 1), files=(Path("a.txt"), Path("b.txt"))),
            options=FeatureSamplingOptions(2),
        )


class TokenNLP:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> NLPDocument:
        self.calls.append(text)
        tokens: list[NLPToken] = []
        offset = 0
        for value in text.split():
            start = text.index(value, offset)
            offset = start + len(value)
            tokens.append(
                NLPToken(
                    value,
                    value.lower(),
                    "PUNCT" if value == "." else "NOUN",
                    start,
                    offset,
                )
            )
        return NLPDocument((NLPSentence(tuple(tokens), text),), text)


def _prepared(text: str) -> PreparedCorpus:
    return PreparedCorpus("g", (Path("a.txt"),), text, text, Counter())


def test_matrix_filters_once_samples_without_repeating_nlp_and_shares_population() -> None:
    backend = TokenNLP()
    rows = build_feature_matrix(
        corpora=(_prepared("aa . bb cc dd ee"),),
        nlp=backend,
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(
            mfw=2,
            field="token",
            sampling=FeatureSamplingOptions(2, 1),
        ),
    )
    assert backend.calls == ["aa . bb cc dd ee"]
    assert len(rows) == 4
    assert tuple(rows[0])[:7] == (
        "group",
        "source_file",
        "sample_id",
        "sample_index",
        "sample_start_token",
        "sample_end_token",
        "sample_kind",
    )
    assert [row["sample_start_token"] for row in rows] == [0, 1, 2, 3]
    assert [row["sample_end_token"] for row in rows] == [2, 3, 4, 5]
    assert all(row["word_token_count"] == 2 for row in rows)
    assert rows[0]["token_count"] == 3
    assert rows[0]["upos_NOUN_count"] == 2
    assert {key for key in rows[0] if key.startswith("mfw_")} == {"mfw_aa", "mfw_bb"}
    assert rows[0]["mfw_aa"] == 0.5
    assert rows[0]["mfw_bb"] == 0.5


def test_mfw_columns_are_selected_before_overlap_sampling() -> None:
    # Unsampled counts select zeta (2); concatenated overlap windows would tie
    # zeta with central alpha (3 weighted occurrences each) and select alpha.
    corpus = (_prepared("zeta zeta alpha beta gamma"),)
    columns = []
    for step in (1, 2):
        rows = build_feature_matrix(
            corpora=corpus,
            nlp=TokenNLP(),
            extraction_policy=AnalysisExtractionPolicy(),
            options=FeatureOptions(
                mfw=1,
                field="token",
                sampling=FeatureSamplingOptions(3, step, include_partial=True),
            ),
        )
        columns.append({key for key in rows[0] if key.startswith("mfw_")})
    assert columns == [{"mfw_zeta"}] * 2


def test_window_population_uses_minimum_length_and_roman_filter() -> None:
    rows = build_feature_matrix(
        corpora=(_prepared("xiv a rosa ."),),
        nlp=TokenNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(
            filter_policy=FeatureFilterPolicy(
                min_token_length=2,
                drop_roman_numerals=True,
            ),
            sampling=FeatureSamplingOptions(1),
        ),
    )
    assert len(rows) == 1
    assert rows[0]["word_token_count"] == 1
    assert rows[0]["sample_start_token"] == 0
    assert rows[0]["sample_end_token"] == 1


def test_short_corpus_errors_without_partial_and_emits_one_with_partial() -> None:
    arguments = dict(
        corpora=(_prepared("aa bb"),),
        nlp=TokenNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
    )
    with pytest.raises(FeatureError, match="produced no samples"):
        build_feature_matrix(
            **arguments,
            options=FeatureOptions(sampling=FeatureSamplingOptions(3)),
        )
    rows = build_feature_matrix(
        **arguments,
        options=FeatureOptions(
            sampling=FeatureSamplingOptions(3, include_partial=True)
        ),
    )
    assert len(rows) == 1
    assert rows[0]["sample_kind"] == "partial"
    assert rows[0]["word_token_count"] == 2


def test_sampling_disabled_keeps_existing_schema() -> None:
    row = build_feature_matrix(
        corpora=(_prepared("aa bb"),),
        nlp=TokenNLP(),
        extraction_policy=AnalysisExtractionPolicy(),
        options=FeatureOptions(),
    )[0]
    assert "source_file" not in row
    assert "sample_id" not in row
    assert row["char_count"] == len("aa bb")
    assert row["token_count"] == 2
