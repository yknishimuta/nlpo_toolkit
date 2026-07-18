from pathlib import Path

import pytest

from nlpo_toolkit.latin.latin_wordlist.collectors import (
    collect_conllu_candidates,
    collect_extra_wordlist_candidates,
    collect_text_candidates,
)
from nlpo_toolkit.latin.latin_wordlist.errors import LatinWordlistSourceReadError
from nlpo_toolkit.latin.latin_wordlist.models import (
    WordlistNoticeCode,
    WordlistTokenizationPolicy,
)
from nlpo_toolkit.latin.latin_wordlist import collectors as collectors_module


def test_conllu_collects_lemmas_and_forms_in_one_pass(tmp_path: Path) -> None:
    source = tmp_path / "tree"
    source.mkdir()
    (source / "b.conllu").write_text("1\tRosa\trosa\nshort\n", encoding="utf-8")
    (source / "a.conllu").write_text("# comment\n1\tRosa\tflos\n\n", encoding="utf-8")
    result, notices = collect_conllu_candidates(directory=source, min_length=2)
    assert [path.name for path in result.files] == ["a.conllu", "b.conllu"]
    assert result.lemmas == {"flos", "rosa"}
    assert result.form_counts == {"rosa": 2}
    assert result.ignored_rows == 1
    assert notices == ()


def test_text_and_extra_collectors(tmp_path: Path) -> None:
    text_dir = tmp_path / "texts"
    text_dir.mkdir()
    (text_dir / "b.txt").write_text("Rosa rosa", encoding="utf-8")
    (text_dir / "a.txt").write_text("AMAT!", encoding="utf-8")
    text, _ = collect_text_candidates(
        directory=text_dir, policy=WordlistTokenizationPolicy(""), min_length=2
    )
    assert [path.name for path in text.files] == ["a.txt", "b.txt"]
    assert text.form_counts == {"amat": 1, "rosa": 2}

    extra_path = tmp_path / "extra.txt"
    extra_path.write_text("# comment\n Homo \nhomo\n123\n", encoding="utf-8")
    extra, _ = collect_extra_wordlist_candidates(path=extra_path)
    assert extra is not None and extra.entries == {"homo"}


def test_missing_sources_return_typed_notices(tmp_path: Path, capsys) -> None:
    conllu, conllu_notices = collect_conllu_candidates(
        directory=tmp_path / "missing-tree", min_length=2
    )
    text, text_notices = collect_text_candidates(
        directory=tmp_path / "missing-text",
        policy=WordlistTokenizationPolicy(""),
        min_length=2,
    )
    extra, extra_notices = collect_extra_wordlist_candidates(path=tmp_path / "missing")
    assert not conllu.files and not text.files and extra is None
    assert conllu_notices[0].code is WordlistNoticeCode.MISSING_CONLLU_DIRECTORY
    assert text_notices[0].code is WordlistNoticeCode.MISSING_TEXT_DIRECTORY
    assert extra_notices[0].code is WordlistNoticeCode.MISSING_EXTRA_WORDLIST
    assert capsys.readouterr() == ("", "")


def test_invalid_utf8_is_a_typed_read_error(tmp_path: Path) -> None:
    source = tmp_path / "texts"
    source.mkdir()
    (source / "bad.txt").write_bytes(b"\xff")
    with pytest.raises(LatinWordlistSourceReadError, match="bad.txt"):
        collect_text_candidates(
            directory=source, policy=WordlistTokenizationPolicy(""), min_length=2
        )


def test_directory_enumeration_failure_is_a_typed_read_error(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "texts"
    source.mkdir()

    def fail_rglob(path, pattern):
        raise PermissionError("denied")

    monkeypatch.setattr(collectors_module.Path, "rglob", fail_rglob)
    with pytest.raises(LatinWordlistSourceReadError, match="denied"):
        collect_text_candidates(
            directory=source, policy=WordlistTokenizationPolicy(""), min_length=2
        )
