from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.nlp import roman_numerals
from nlpo_toolkit.nlp.roman_numerals import (
    RomanExceptionsError,
    effective_roman_exceptions,
    load_roman_exceptions,
    normalize_roman_exceptions,
    should_drop_roman_numeral,
)


def test_loader_normalizes_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "roman.txt"
    path.write_text("# comment\n\n XIV \nvi\nxiv\n", encoding="utf-8")
    assert load_roman_exceptions(path) == frozenset({"xiv", "vi"})


def test_loader_reports_missing_directory_and_invalid_utf8(tmp_path: Path) -> None:
    with pytest.raises(RomanExceptionsError, match=str(tmp_path / "missing")):
        load_roman_exceptions(tmp_path / "missing")
    with pytest.raises(RomanExceptionsError, match="must be a file"):
        load_roman_exceptions(tmp_path)
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff")
    with pytest.raises(RomanExceptionsError, match=str(invalid)):
        load_roman_exceptions(invalid)


def test_exception_policy_is_normalized_and_does_not_mutate_input() -> None:
    configured = [" XIV ", ""]
    assert normalize_roman_exceptions(configured) == frozenset({"xiv"})
    assert effective_roman_exceptions(
        use_lemma=True, configured_exceptions=configured
    ) == frozenset({"xiv"})
    assert effective_roman_exceptions(
        use_lemma=False, configured_exceptions=configured
    ) == frozenset({"xiv", "vi", "di"})
    assert configured == [" XIV ", ""]


def test_drop_policy_handles_case_exceptions_and_normal_words() -> None:
    assert not should_drop_roman_numeral(
        "XIV", drop_roman_numerals=False, effective_exceptions=()
    )
    assert should_drop_roman_numeral(
        "XIV", drop_roman_numerals=True, effective_exceptions=()
    )
    assert not should_drop_roman_numeral(
        "XIV", drop_roman_numerals=True, effective_exceptions={"xiv"}
    )
    assert not should_drop_roman_numeral(
        "rosa", drop_roman_numerals=True, effective_exceptions=()
    )


def test_mixed_resolver_api_is_absent() -> None:
    assert not hasattr(roman_numerals, "resolve_roman_exceptions")
