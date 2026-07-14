from __future__ import annotations

import pytest

from nlpo_toolkit.nlp.chunking import iter_char_chunks


def test_empty_and_short_text() -> None:
    assert list(iter_char_chunks("", 3)) == []
    assert list(iter_char_chunks("abc", 3)) == ["abc"]


def test_chunks_preserve_text_and_prefer_whitespace() -> None:
    text = "arma virumque cano"
    chunks = list(iter_char_chunks(text, 8))
    assert chunks[0] == "arma"
    assert "".join(chunks) == text


def test_massive_word_uses_safe_fixed_width_fallback() -> None:
    assert list(iter_char_chunks("abcdefghij", 4)) == ["abcd", "efgh", "ij"]


def test_unicode_text_is_not_lost_or_duplicated() -> None:
    text = "Æneas rōmam amat"
    assert "".join(iter_char_chunks(text, 6)) == text


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_chunk_size_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        list(iter_char_chunks("abc", value))  # type: ignore[arg-type]
