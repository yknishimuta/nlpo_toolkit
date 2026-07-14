from __future__ import annotations

from tests.corpus_analysis.fake_nlp import corpus_request

import json
from pathlib import Path

import pytest

import nlpo_toolkit.corpus_analysis.analysis_pipeline as analysis_pipeline_mod
import nlpo_toolkit.corpus_analysis.run_reporting as run_reporting_mod
import nlpo_toolkit.corpus_analysis.runner as runner_mod
from tests.corpus_analysis.fake_nlp import fake_backend_factory, runner_dependencies


class NoopCleaner:
    @staticmethod
    def main(_argv):
        return 0


def _write_cleaner_config(root: Path) -> None:
    (root / "input").mkdir(exist_ok=True)
    (root / "cleaner.yml").write_text(
        "kind: scholastic_text\ninput: input\noutput: cleaned\n",
        encoding="utf-8",
    )

def test_run_minimal_success(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")
    (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
    (tmp_path / "b.txt").write_text("BBB", encoding="utf-8")

    # --- fake config
    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "nlp": {
                "language": "la",
                "stanza_package": "perseus",
                "cpu_only": True,
            },
            "groups": {
                "g1": {"files": ["a.txt"]},
                "g2": {"files": ["b.txt"]},
            },
        }

    # --- stub preprocess

    # --- capture CSV writes
    calls = []
    def write_frequency_csv(path, counter, header):
        calls.append((Path(path), dict(counter), header))

    monkeypatch.setattr(analysis_pipeline_mod, "write_frequency_csv", write_frequency_csv)

    # --- meta
    monkeypatch.setattr(run_reporting_mod, "build_run_meta", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(run_reporting_mod, "collect_runtime_environment", lambda _sd: {"py": "x"})
    meta_calls = []
    monkeypatch.setattr(run_reporting_mod, "write_run_meta", lambda meta, out_dir: meta_calls.append((meta, Path(out_dir))))

    rc = runner_mod.run(
        corpus_request(script_dir, config_path),
        dependencies=runner_dependencies(
            load_config_fn,
            fake_backend_factory(
                [("deus", "deus", "NOUN"), ("deus", "deus", "NOUN"), ("angelus", "angelus", "NOUN")]
            ),
        ),
    )
    assert rc.exit_code == 0

    out_dir = tmp_path / "output"
    assert (out_dir / "summary.txt").exists()

    # csv
    assert len(calls) == 2
    assert calls[0][0].name == "frequency_g1.csv"
    assert calls[1][0].name == "frequency_g2.csv"

    # meta
    assert len(meta_calls) == 1
    meta, meta_out_dir = meta_calls[0]
    assert meta_out_dir == out_dir
    assert meta["analysis_unit"] == "lemma"
    assert meta["environment"] == {"py": "x"}

def test_run_analysis_unit_surface_uses_surface_form(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")
    (tmp_path / "a.txt").write_text("X", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "analysis_unit": "surface",
            "groups": {"g": {"files": ["a.txt"]}},
        }


    monkeypatch.setattr(run_reporting_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(run_reporting_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(run_reporting_mod, "write_run_meta", lambda meta, out_dir: None)

    calls = []
    monkeypatch.setattr(
            analysis_pipeline_mod,
            "write_frequency_csv",
        lambda path, counter, header: calls.append(dict(counter)),
    )

    rc = runner_mod.run(
        corpus_request(script_dir, config_path),
        dependencies=runner_dependencies(
            load_config_fn,
            fake_backend_factory([("X", "lemma_x", "NOUN")]),
        ),
    )
    assert rc.exit_code == 0
    assert calls == [{"x": 1}]

def test_run_dictcheck_requires_wordlist(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")
    (tmp_path / "a.txt").write_text("X", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"g": {"files": ["a.txt"]}},
            "dictcheck": {"enabled": True},
        }

    monkeypatch.setattr(run_reporting_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(run_reporting_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(run_reporting_mod, "write_run_meta", lambda meta, out_dir: None)

    with pytest.raises(ValueError):
        runner_mod.run(
            corpus_request(script_dir, config_path),
            dependencies=runner_dependencies(
                load_config_fn,
                fake_backend_factory([("a", "a", "NOUN")]),
            ),
        )


def test_run_dictcheck_writes_known_unknown(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")
    (tmp_path / "a.txt").write_text("X", encoding="utf-8")

    wl = tmp_path / "wordlist.txt"
    wl.write_text("deus\n", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"g": {"files": ["a.txt"]}},
            "dictcheck": {"enabled": True, "wordlist": str(wl)},
        }

    monkeypatch.setattr(run_reporting_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(run_reporting_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(run_reporting_mod, "write_run_meta", lambda meta, out_dir: None)

    calls = []
    monkeypatch.setattr(analysis_pipeline_mod, "write_frequency_csv",
                        lambda path, c, header: calls.append(Path(path).name))

    rc = runner_mod.run(
        corpus_request(script_dir, config_path),
        dependencies=runner_dependencies(
            load_config_fn,
            fake_backend_factory(
                [("deus", "deus", "NOUN"), ("deus", "deus", "NOUN"), ("angelus", "angelus", "NOUN")]
            ),
        ),
    )
    assert rc.exit_code == 0

    assert calls == [
        "frequency_g.csv",
        "frequency_g.known.csv",
        "frequency_g.unknown.csv",
    ]

def test_run_applies_filter_options_in_record_pipeline(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")
    (tmp_path / "a.txt").write_text("X", encoding="utf-8")

    exc_file = tmp_path / "rom.txt"
    exc_file.write_text("xiv\n", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "analysis_unit": "lemma",
            "groups": {"g": {"files": ["a.txt"]}},
            "filters": {
                "min_token_length": 3,
                "drop_roman_numerals": True,
                "roman_exceptions_file": "rom.txt"
            }
        }


    monkeypatch.setattr(run_reporting_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(run_reporting_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(run_reporting_mod, "write_run_meta", lambda meta, out_dir: None)
    counters = []
    monkeypatch.setattr(
            analysis_pipeline_mod,
            "write_frequency_csv",
        lambda path, counter, header: counters.append(dict(counter)),
    )

    rc = runner_mod.run(
        corpus_request(script_dir, config_path),
        dependencies=runner_dependencies(
            load_config_fn,
            fake_backend_factory(
                [("ab", "ab", "NOUN"), ("xiv", "xiv", "NOUN"), ("rosa", "rosa", "NOUN")]
            ),
        ),
    )
    
    assert rc.exit_code == 0
    assert counters == [{"xiv": 1, "rosa": 1}]


def test_run_group_by_file_writes_one_csv_per_input_file(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "virgil-aeneis.txt").write_text("arma virumque", encoding="utf-8")
    (input_dir / "file2.txt").write_text("cano", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"all": {"files": ["input/*.txt"]}},
            "grouping": {"mode": "per_file"},
        }

    monkeypatch.setattr(run_reporting_mod, "build_run_meta", lambda **kwargs: {"groups_files": kwargs["groups_files"]})
    monkeypatch.setattr(run_reporting_mod, "collect_runtime_environment", lambda _sd: {})

    csv_calls = []
    monkeypatch.setattr(
            analysis_pipeline_mod,
            "write_frequency_csv",
        lambda path, c, header: csv_calls.append((Path(path).name, dict(c))),
    )

    meta_calls = []
    monkeypatch.setattr(run_reporting_mod, "write_run_meta", lambda meta, out_dir: meta_calls.append(meta))

    rc = runner_mod.run(
        corpus_request(project_root, config_path),
        dependencies=runner_dependencies(load_config_fn, fake_backend_factory()),
    )

    assert rc.exit_code == 0
    assert [name for name, _counter in csv_calls] == [
        "frequency_file2.csv",
        "frequency_virgil_aeneis.csv",
    ]
    assert meta_calls[0]["grouping"] == {"mode": "per_file"}
    assert sorted(meta_calls[0]["groups_files"]) == ["file2", "virgil_aeneis"]


def test_run_group_by_file_cli_override(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "file3.txt").write_text("rosa", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"all": {"files": ["input/*.txt"]}},
        }

    monkeypatch.setattr(run_reporting_mod, "build_run_meta", lambda **kwargs: {"groups_files": kwargs["groups_files"]})
    monkeypatch.setattr(run_reporting_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(run_reporting_mod, "write_run_meta", lambda meta, out_dir: None)

    csv_names = []
    monkeypatch.setattr(
            analysis_pipeline_mod,
            "write_frequency_csv",
        lambda path, c, header: csv_names.append(Path(path).name),
    )

    rc = runner_mod.run(
        corpus_request(project_root, config_path, group_by_file=True),
        dependencies=runner_dependencies(load_config_fn, fake_backend_factory()),
    )

    assert rc.exit_code == 0
    assert csv_names == ["frequency_file3.csv"]


def test_run_meta_records_generated_outputs_and_actual_group_files(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    used = input_dir / "used.txt"
    used.write_text("rosa", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"text": {"files": ["input/*.txt"]}},
            "dictcheck": {"enabled": False},
        }


    rc = runner_mod.run(
        corpus_request(project_root, config_path),
        dependencies=runner_dependencies(load_config_fn, fake_backend_factory()),
    )

    assert rc.exit_code == 0
    meta = json.loads((project_root / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["groups_files"] == {"text": [str(used.resolve())]}
    generated_names = {Path(p).name for p in meta["generated_outputs"]}
    assert generated_names == {"frequency_text.csv", "summary.txt", "run_meta.json"}


def test_run_error_on_empty_group(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {"out_dir": "output", "groups": {"empty": {"files": ["input/*.txt"]}}}


    with pytest.raises(ValueError, match="No files matched"):
        runner_mod.run(
            corpus_request(project_root, config_path, error_on_empty_group=True),
            dependencies=runner_dependencies(load_config_fn, fake_backend_factory()),
        )


def test_run_auto_single_cleaned_records_selected_file(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    selected = cleaned_dir / "satyricon.cleaned.txt"
    selected.write_text("rosa", encoding="utf-8")
    _write_cleaner_config(tmp_path)

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "preprocess": {"kind": "cleaner", "config": "cleaner.yml"},
            "groups": {"old": {"files": ["cleaned/*.txt"]}},
            "grouping": {"mode": "auto_single_cleaned", "auto_group_name": "text"},
        }

    rc = runner_mod.run(
        corpus_request(project_root, config_path),
        dependencies=runner_dependencies(
            load_config_fn,
            fake_backend_factory(),
            cleaner=NoopCleaner(),
        ),
    )

    assert rc.exit_code == 0
    meta = json.loads((project_root / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["grouping"] == {"mode": "auto_single_cleaned", "auto_group_name": "text"}
    assert meta["groups_files"] == {"text": [str(selected.resolve())]}


def test_run_auto_single_cleaned_errors_on_zero_files(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    _write_cleaner_config(tmp_path)

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "preprocess": {"kind": "cleaner", "config": "cleaner.yml"},
            "groups": {"old": {"files": ["cleaned/*.txt"]}},
            "grouping": {"mode": "auto_single_cleaned"},
        }

    with pytest.raises(ValueError, match="no \\.txt files"):
        runner_mod.run(
            corpus_request(project_root, config_path),
            dependencies=runner_dependencies(
                load_config_fn,
                fake_backend_factory(),
                cleaner=NoopCleaner(),
            ),
        )


def test_run_auto_single_cleaned_errors_on_multiple_files(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    (cleaned_dir / "a.cleaned.txt").write_text("a", encoding="utf-8")
    (cleaned_dir / "b.cleaned.txt").write_text("b", encoding="utf-8")
    (cleaned_dir / ".DS_Store").write_text("ignored", encoding="utf-8")
    (cleaned_dir / ".gitkeep").write_text("", encoding="utf-8")
    _write_cleaner_config(tmp_path)

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "preprocess": {"kind": "cleaner", "config": "cleaner.yml"},
            "groups": {"old": {"files": ["cleaned/*.txt"]}},
            "grouping": {"mode": "auto_single_cleaned"},
        }

    with pytest.raises(ValueError, match="expected exactly one"):
        runner_mod.run(
            corpus_request(project_root, config_path),
            dependencies=runner_dependencies(
                load_config_fn,
                fake_backend_factory(),
                cleaner=NoopCleaner(),
            ),
        )
