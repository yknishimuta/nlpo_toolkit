from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_records import NLPAnalysisRecord
from nlpo_toolkit.corpus_analysis.features.errors import FeatureError
from nlpo_toolkit.corpus_analysis.features.function_word_loader import (
    load_function_word_vocabulary,
)
from nlpo_toolkit.corpus_analysis.features.function_words import (
    build_function_word_columns,
    compute_function_word_features,
)
from nlpo_toolkit.corpus_analysis.features.models import (
    FunctionWordOptions,
    FunctionWordVocabulary,
)


def test_loader_preserves_order_and_normalizes_comments_and_blank_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "function_words.txt"
    path.write_text("# comment\n  ET  \n\nAutem\nin\n", encoding="utf-8")

    vocabulary = load_function_word_vocabulary(path)

    assert vocabulary.terms == ("et", "autem", "in")
    assert isinstance(vocabulary.terms, tuple)
    with pytest.raises(FrozenInstanceError):
        vocabulary.terms = ()  # type: ignore[misc]


@pytest.mark.parametrize("content", ("# only\n\n", ""))
def test_loader_rejects_lists_without_terms(tmp_path: Path, content: str) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(FeatureError, match="contains no terms"):
        load_function_word_vocabulary(path)


def test_loader_rejects_normalized_duplicate_with_both_line_numbers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.txt"
    path.write_text("et\n# line two\nET\n", encoding="utf-8")
    with pytest.raises(FeatureError, match=r"'et'.*1.*3"):
        load_function_word_vocabulary(path)


@pytest.mark.parametrize("term", ("in quo", "in\tquo"))
def test_loader_rejects_multiword_and_tab_terms(tmp_path: Path, term: str) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text(f"{term}\n", encoding="utf-8")
    with pytest.raises(FeatureError, match="single token.*line 1"):
        load_function_word_vocabulary(path)


def test_loader_reports_missing_and_invalid_utf8_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(FeatureError, match=f"file not found: {missing}"):
        load_function_word_vocabulary(missing)

    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"et\n\xff")
    with pytest.raises(FeatureError, match="not valid UTF-8"):
        load_function_word_vocabulary(invalid)


def test_column_names_preserve_vocabulary_order_and_use_safe_names() -> None:
    vocabulary = FunctionWordVocabulary(("et", "at-que", "non"))
    assert build_function_word_columns(vocabulary) == (
        ("et", "fw_et"),
        ("at-que", "fw_at_que"),
        ("non", "fw_non"),
    )


def test_column_collision_and_empty_suffix_are_errors() -> None:
    with pytest.raises(FeatureError, match=r"collision.*'a-b'.*'a!b'.*fw_a_b"):
        build_function_word_columns(FunctionWordVocabulary(("a-b", "a!b")))
    with pytest.raises(FeatureError, match="empty suffix"):
        build_function_word_columns(FunctionWordVocabulary(("!",)))


def _record(token: str, lemma: str | None, *, upos: str = "NOUN") -> NLPAnalysisRecord:
    return NLPAnalysisRecord(
        chunk_index=0,
        sentence_index=0,
        token_index=0,
        global_token_index=0,
        char_start_in_chunk=None,
        char_end_in_chunk=None,
        char_start_in_text=None,
        char_end_in_text=None,
        sentence="",
        token=token,
        lemma=lemma,
        upos=upos,
    )


def test_calculator_uses_lexical_denominator_and_emits_absent_terms() -> None:
    records = (
        _record(" ET ", "et"),
        _record("rosa", "rosa"),
        _record("et", "et"),
        _record("IN", "in"),
        _record("villa", "villa"),
        _record("est", "sum"),
    )
    vocabulary = FunctionWordVocabulary(("et", "in", "non"))

    row = compute_function_word_features(
        records,
        options=FunctionWordOptions(vocabulary=vocabulary, field="lemma"),
    )

    assert tuple(row) == ("fw_et", "fw_in", "fw_non")
    assert row == {"fw_et": 2 / 6, "fw_in": 1 / 6, "fw_non": 0.0}
    assert all(isinstance(value, float) for value in row.values())
    assert all(0.0 <= value <= 1.0 for value in row.values())
    assert vocabulary.terms == ("et", "in", "non")


def test_empty_records_emit_all_zero_columns() -> None:
    row = compute_function_word_features(
        (),
        options=FunctionWordOptions(FunctionWordVocabulary(("et", "in"))),
    )
    assert row == {"fw_et": 0.0, "fw_in": 0.0}


def test_lemma_token_selection_and_lemma_fallback_are_independent() -> None:
    records = (
        _record(" EST ", "sum"),
        _record("AUTEM", None),
    )
    vocabulary = FunctionWordVocabulary(("sum", "est", "autem"))
    lemma = compute_function_word_features(
        records,
        options=FunctionWordOptions(vocabulary, field="lemma"),
    )
    token = compute_function_word_features(
        records,
        options=FunctionWordOptions(vocabulary, field="token"),
    )

    assert lemma == {"fw_sum": 0.5, "fw_est": 0.0, "fw_autem": 0.5}
    assert token == {"fw_sum": 0.0, "fw_est": 0.5, "fw_autem": 0.5}


def test_explicit_matching_does_not_depend_on_upos_category() -> None:
    record = _record("et", "et", upos="NOUN")
    assert compute_function_word_features(
        (record,),
        options=FunctionWordOptions(FunctionWordVocabulary(("et",))),
    ) == {"fw_et": 1.0}
