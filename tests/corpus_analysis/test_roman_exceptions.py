from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

import nlpo_toolkit.corpus_analysis.nlp_hooks as nlp_hooks
import nlpo_toolkit.corpus_analysis.runner as runner_mod
from nlpo_toolkit.nlp import (
    RomanExceptionsError,
    count_nouns,
    count_nouns_streaming,
    load_roman_exceptions,
)


@dataclass
class Token:
    text: str
    lemma: str
    upos: str = "NOUN"
    start_char: int = 0


@dataclass
class Sentence:
    tokens: list[Token]
    text: str = ""

    @property
    def words(self) -> list[Token]:
        return self.tokens


@dataclass
class Doc:
    sentences: list[Sentence]


class FakeNLP:
    def __init__(self, tokens: list[Token]):
        self.doc = Doc([Sentence(tokens=tokens, text=" ".join(t.text for t in tokens))])

    def __call__(self, text: str) -> Doc:
        return self.doc


def _fake_nlp() -> FakeNLP:
    return FakeNLP(
        [
            Token(text="xiv", lemma="xiv", start_char=0),
            Token(text="iv", lemma="iv", start_char=4),
            Token(text="rosa", lemma="rosa", start_char=7),
        ]
    )


def test_load_roman_exceptions_ignores_blank_lines_comments_and_duplicates(tmp_path: Path):
    path = tmp_path / "roman.txt"
    path.write_text("# comment\n\nXIV\nvi\nxiv\n", encoding="utf-8")

    assert load_roman_exceptions(path) == frozenset({"xiv", "vi"})


def test_load_roman_exceptions_errors_for_missing_file(tmp_path: Path):
    with pytest.raises(RomanExceptionsError, match="was not found"):
        load_roman_exceptions(tmp_path / "missing.txt")


def test_load_roman_exceptions_errors_for_directory(tmp_path: Path):
    with pytest.raises(RomanExceptionsError, match="must be a file"):
        load_roman_exceptions(tmp_path)


def test_count_nouns_uses_configured_roman_exception_file(tmp_path: Path):
    path = tmp_path / "roman.txt"
    path.write_text("xiv\n", encoding="utf-8")

    result = count_nouns(
        "ignored",
        _fake_nlp(),
        use_lemma=False,
        drop_roman_numerals=True,
        roman_exceptions_file=path,
    )

    assert result["xiv"] == 1
    assert "iv" not in result
    assert result["rosa"] == 1


def test_count_nouns_exception_file_is_case_insensitive(tmp_path: Path):
    path = tmp_path / "roman.txt"
    path.write_text("XIV\n", encoding="utf-8")

    result = count_nouns(
        "ignored",
        _fake_nlp(),
        use_lemma=False,
        drop_roman_numerals=True,
        roman_exceptions_file=path,
    )

    assert result["xiv"] == 1


def test_surface_mode_keeps_builtin_and_configured_exceptions(tmp_path: Path):
    path = tmp_path / "roman.txt"
    path.write_text("xiv\n", encoding="utf-8")
    nlp = FakeNLP(
        [
            Token(text="vi", lemma="vi"),
            Token(text="xiv", lemma="xiv"),
            Token(text="iv", lemma="iv"),
        ]
    )

    result = count_nouns(
        "ignored",
        nlp,
        use_lemma=False,
        drop_roman_numerals=True,
        roman_exceptions_file=path,
    )

    assert result == Counter({"vi": 1, "xiv": 1})


def test_lemma_mode_uses_configured_exceptions_but_not_surface_builtin(tmp_path: Path):
    path = tmp_path / "roman.txt"
    path.write_text("xiv\n", encoding="utf-8")
    nlp = FakeNLP(
        [
            Token(text="vi", lemma="vi"),
            Token(text="xiv", lemma="xiv"),
            Token(text="iv", lemma="iv"),
        ]
    )

    result = count_nouns(
        "ignored",
        nlp,
        use_lemma=True,
        drop_roman_numerals=True,
        roman_exceptions_file=path,
    )

    assert result == Counter({"xiv": 1})


def test_fast_and_trace_paths_use_same_roman_exceptions(tmp_path: Path):
    trace_path = tmp_path / "trace.tsv"

    fast = count_nouns_streaming(
        "ignored",
        _fake_nlp(),
        use_lemma=False,
        drop_roman_numerals=True,
        roman_exceptions=frozenset({"xiv"}),
    )
    traced = count_nouns_streaming(
        "ignored",
        _fake_nlp(),
        use_lemma=False,
        drop_roman_numerals=True,
        roman_exceptions=frozenset({"xiv"}),
        trace_tsv=trace_path,
        trace_max_rows=100,
        label="group_a",
    )

    assert fast == traced == Counter({"xiv": 1, "rosa": 1})
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "xiv" in trace_text
    assert "\tiv\t" not in trace_text


def test_nlp_hooks_count_group_uses_roman_exception_file(tmp_path: Path):
    path = tmp_path / "roman.txt"
    path.write_text("xiv\n", encoding="utf-8")

    result = nlp_hooks.count_group(
        "ignored",
        _fake_nlp(),
        use_lemma=False,
        drop_roman_numerals=True,
        roman_exceptions_file=path,
    )

    assert result == Counter({"xiv": 1, "rosa": 1})


def test_runner_integration_uses_roman_exception_file_in_final_csv(tmp_path: Path, monkeypatch):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "roman.txt").write_text("xiv\n", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config_fn(_path: Path):
        return {
            "out_dir": "output",
            "analysis_unit": "surface",
            "groups": {"group_a": {"files": ["input/a.txt"]}},
            "filters": {
                "drop_roman_numerals": True,
                "roman_exceptions_file": "config/roman.txt",
            },
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)

    rc = runner_mod.run(
        project_root=tmp_path,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *args, **kwargs: (_fake_nlp(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=nlp_hooks.count_group,
        render_stanza_package_table_fn=lambda *args, **kwargs: [],
    )

    assert rc == 0
    rows = list(
        csv.DictReader(
            (tmp_path / "output" / "frequency_group_a.csv").open(
                encoding="utf-8"
            )
        )
    )
    counts = {row["word"]: int(row["frequency"]) for row in rows}
    assert counts == {"xiv": 1, "rosa": 1}


def test_runner_loads_roman_exceptions_once_for_multiple_groups(tmp_path: Path, monkeypatch):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "input" / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "roman.txt").write_text("xiv\n", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config_fn(_path: Path):
        return {
            "out_dir": "output",
            "groups": {
                "group_a": {"files": ["input/a.txt"]},
                "group_b": {"files": ["input/b.txt"]},
            },
            "filters": {
                "drop_roman_numerals": True,
                "roman_exceptions_file": "config/roman.txt",
            },
        }

    calls: list[Path] = []

    def fake_loader(path: Path) -> frozenset[str]:
        calls.append(path)
        return frozenset({"xiv"})

    seen_exceptions: list[frozenset[str]] = []

    def fake_count_group(text, nlp, **kwargs):
        seen_exceptions.append(kwargs["roman_exceptions"])
        return Counter({"xiv": 1})

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "load_roman_exceptions", fake_loader)

    rc = runner_mod.run(
        project_root=tmp_path,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *args, **kwargs: (object(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=fake_count_group,
        render_stanza_package_table_fn=lambda *args, **kwargs: [],
    )

    assert rc == 0
    assert calls == [(tmp_path / "config" / "roman.txt").resolve()]
    assert seen_exceptions == [frozenset({"xiv"}), frozenset({"xiv"})]
