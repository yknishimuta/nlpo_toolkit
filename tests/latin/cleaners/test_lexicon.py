from pathlib import Path

import pytest

from nlpo_toolkit.latin.cleaners.errors import CleanerLexiconError
from nlpo_toolkit.latin.cleaners.lexicon import apply_lexicon_map, load_lexicon_map


def test_lexicon_load_is_immutable_and_application_uses_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.tsv"
    path.write_text("# comment\n\nfoo\tfirst\nfoo\tbar\nfoobar\tbaz\n", encoding="utf-8")
    mapping = load_lexicon_map(path)
    assert apply_lexicon_map("foo foobar food", mapping) == "bar baz food"
    with pytest.raises(TypeError):
        mapping["x"] = "y"  # type: ignore[index]


def test_lexicon_bad_row_reports_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("valid\trow\nbad\n", encoding="utf-8")
    with pytest.raises(CleanerLexiconError, match=r"bad\.tsv:2"):
        load_lexicon_map(path)
