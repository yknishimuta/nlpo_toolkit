from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.corpus_errors import CorpusReadError
from nlpo_toolkit.corpus_analysis.io_utils import read_concat


def test_read_concat_preserves_order_and_separator(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    assert read_concat((first, second)) == "alpha\nbeta"
    assert read_concat((second, first)) == "beta\nalpha"


def test_read_concat_empty_sequence_returns_empty_text() -> None:
    assert read_concat(()) == ""


def test_read_concat_fails_on_missing_file(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    missing = tmp_path / "missing.txt"
    existing.write_text("alpha", encoding="utf-8")

    with pytest.raises(CorpusReadError, match="missing.txt") as exc_info:
        read_concat((existing, missing))

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_read_concat_stops_at_first_failure(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    third = tmp_path / "third.txt"
    first.write_text("alpha", encoding="utf-8")
    calls: list[Path] = []
    original = Path.read_text

    def recording_read_text(self: Path, *args, **kwargs):
        calls.append(self)
        if self == second:
            raise PermissionError("denied")
        if self == third:
            raise AssertionError("third file must not be read")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    with pytest.raises(CorpusReadError, match="second.txt") as exc_info:
        read_concat((first, second, third))

    assert isinstance(exc_info.value.__cause__, PermissionError)
    assert calls == [first, second]


def test_read_concat_fails_on_invalid_utf8(tmp_path: Path) -> None:
    broken = tmp_path / "broken.txt"
    broken.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(CorpusReadError, match="broken.txt") as exc_info:
        read_concat((broken,))

    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_read_concat_fails_when_path_is_directory(tmp_path: Path) -> None:
    directory = tmp_path / "corpus"
    directory.mkdir()

    with pytest.raises(CorpusReadError, match="corpus") as exc_info:
        read_concat((directory,))

    assert isinstance(exc_info.value.__cause__, OSError)
