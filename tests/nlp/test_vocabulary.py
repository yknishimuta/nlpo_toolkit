from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.nlp.vocabulary import load_wordlist


def test_wordlist_is_trimmed_deduplicated_immutable_and_not_normalized(
    tmp_path: Path,
) -> None:
    path = tmp_path / "words.txt"
    path.write_text("# comment\n\n Rosa \nRosa\nrosa\nrōsa\n", encoding="utf-8")
    result = load_wordlist(path)
    assert result == frozenset({"Rosa", "rosa", "rōsa"})
    assert isinstance(result, frozenset)


def test_wordlist_preserves_file_errors(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_wordlist(tmp_path / "missing.txt")
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff")
    with pytest.raises(UnicodeError):
        load_wordlist(invalid)
