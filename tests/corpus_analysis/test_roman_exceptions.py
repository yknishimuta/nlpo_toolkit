from __future__ import annotations

import csv
from pathlib import Path

import pytest

import nlpo_toolkit.corpus_analysis.runner as runner_mod
import nlpo_toolkit.corpus_analysis.runtime as runtime_mod
from nlpo_toolkit.nlp import (
    RomanExceptionsError,
    load_roman_exceptions,
)
from tests.corpus_analysis.fake_nlp import fake_backend_factory, runner_dependencies


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


    rc = runner_mod.run(
        project_root=tmp_path,
        config_path=config_path,
        dependencies=runner_dependencies(
            load_config_fn,
            fake_backend_factory(
                [("xiv", "xiv", "NOUN"), ("iv", "iv", "NOUN"), ("rosa", "rosa", "NOUN")]
            ),
        ),
    )

    assert rc.exit_code == 0
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

    monkeypatch.setattr(runtime_mod, "load_roman_exceptions", fake_loader)

    rc = runner_mod.run(
        project_root=tmp_path,
        config_path=config_path,
        dependencies=runner_dependencies(
            load_config_fn,
            fake_backend_factory([("xiv", "xiv", "NOUN"), ("iv", "iv", "NOUN")]),
        ),
    )

    assert rc.exit_code == 0
    assert calls == [(tmp_path / "config" / "roman.txt").resolve()]
    for label in ("group_a", "group_b"):
        rows = list(
            csv.DictReader(
                (tmp_path / "output" / f"frequency_{label}.csv").open(
                    encoding="utf-8"
                )
            )
        )
        assert {row["lemma"]: int(row["count"]) for row in rows} == {"xiv": 1}
