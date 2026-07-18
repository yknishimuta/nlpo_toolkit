from pathlib import Path

import pytest

from nlpo_toolkit.latin.latin_wordlist.errors import LatinWordlistPublicationError
from nlpo_toolkit.latin.latin_wordlist.models import WordlistPublication
from nlpo_toolkit.latin.latin_wordlist.publication import publish_wordlist
from nlpo_toolkit.latin.latin_wordlist import publication as publication_module


def test_publication_writes_utf8_lines_and_replaces_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "words.txt"
    output.parent.mkdir()
    output.write_text("old\n", encoding="utf-8")
    publish_wordlist(WordlistPublication(output, ("amo", "rōsa")))
    assert output.read_text(encoding="utf-8") == "amo\nrōsa\n"
    assert not tuple(output.parent.glob("*.tmp"))


def test_empty_publication_and_directory_error(tmp_path: Path) -> None:
    output = tmp_path / "empty.txt"
    publish_wordlist(WordlistPublication(output, ()))
    assert output.read_bytes() == b""
    with pytest.raises(LatinWordlistPublicationError):
        publish_wordlist(WordlistPublication(tmp_path, ()))


def test_failed_atomic_replace_preserves_existing_output(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "words.txt"
    output.write_text("old\n", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(publication_module.os, "replace", fail_replace)
    with pytest.raises(LatinWordlistPublicationError, match="replace failed"):
        publish_wordlist(WordlistPublication(output, ("new",)))
    assert output.read_text(encoding="utf-8") == "old\n"
    assert not tuple(tmp_path.glob("*.tmp"))
