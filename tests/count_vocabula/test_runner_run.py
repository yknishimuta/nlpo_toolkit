from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

import nlpo_toolkit.count_vocabula.runner as runner_mod

def test_run_minimal_success(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    # --- fake config
    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "language": "la",
            "stanza_package": "perseus",
            "cpu_only": True,
            "groups": {
                "g1": {"files": ["a.txt"]},
                "g2": {"files": ["b.txt"]},
            },
        }

    # --- stub preprocess
    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "expand_cleaned_dir_placeholders", lambda patterns, cleaned_dir: patterns)

    # --- stub IO
    monkeypatch.setattr(runner_mod, "expand_globs", lambda patterns: [Path(p) for p in patterns])
    monkeypatch.setattr(runner_mod, "read_concat", lambda files: "AAA BBB")

    # --- stub NLP & counter
    def build_pipeline_fn(language, stanza_package, cpu_only):
        return object(), stanza_package

    def count_group_fn(text, nlp, use_lemma=True, upos_targets=None, **kwargs):
        assert text  # joined
        return Counter({"deus": 2, "angelus": 1})

    # --- capture CSV writes
    calls = []
    def write_frequency_csv(path, counter, header):
        calls.append((Path(path), dict(counter), header))

    monkeypatch.setattr(runner_mod, "write_frequency_csv", write_frequency_csv)

    # --- meta
    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {"py": "x"})
    meta_calls = []
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: meta_calls.append((meta, Path(out_dir))))

    # --- stanza package table
    def render_stanza_package_table_fn(nlp, stanza_package):
        return ["table: ok"]

    rc = runner_mod.run(
        script_dir=script_dir,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=build_pipeline_fn,
        build_sentence_splitter_fn=None,
        count_group_fn=count_group_fn,
        render_stanza_package_table_fn=render_stanza_package_table_fn,
    )
    assert rc == 0

    out_dir = tmp_path / "output"
    assert (out_dir / "summary.txt").exists()

    # csv
    assert len(calls) == 2
    assert calls[0][0].name == "noun_frequency_g1.csv"
    assert calls[1][0].name == "noun_frequency_g2.csv"

    # meta
    assert len(meta_calls) == 1
    meta, meta_out_dir = meta_calls[0]
    assert meta_out_dir == out_dir
    assert meta["analysis_unit"] == "lemma"
    assert meta["environment"] == {"py": "x"}

def test_run_analysis_unit_surface_passes_use_lemma_false(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "analysis_unit": "surface",
            "groups": {"g": {"files": ["a.txt"]}},
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "expand_cleaned_dir_placeholders", lambda patterns, cleaned_dir: patterns)
    monkeypatch.setattr(runner_mod, "expand_globs", lambda patterns: [Path("a.txt")])
    monkeypatch.setattr(runner_mod, "read_concat", lambda files: "X")

    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: None)

    monkeypatch.setattr(runner_mod, "write_frequency_csv", lambda *a, **k: None)

    seen = {}
    def count_group_fn(text, nlp, use_lemma=True, **kwargs):
        seen["use_lemma"] = use_lemma
        return Counter({"X": 1})

    rc = runner_mod.run(
        script_dir=script_dir,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=count_group_fn,
        render_stanza_package_table_fn=lambda *a, **k: [],
    )
    assert rc == 0
    assert seen["use_lemma"] is False

def test_run_dictcheck_requires_wordlist(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"g": {"files": ["a.txt"]}},
            "dictcheck": {"enabled": True},
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "expand_cleaned_dir_placeholders", lambda patterns, cleaned_dir: patterns)
    monkeypatch.setattr(runner_mod, "expand_globs", lambda patterns: [Path("a.txt")])
    monkeypatch.setattr(runner_mod, "read_concat", lambda files: "X")
    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: None)

    with pytest.raises(ValueError):
        runner_mod.run(
            script_dir=script_dir,
            config_path=config_path,
            load_config_fn=load_config_fn,
            clean_mod=object(),
            build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
            build_sentence_splitter_fn=None,
            count_group_fn=lambda *a, **k: Counter({"a": 1}),
            render_stanza_package_table_fn=lambda *a, **k: [],
        )


def test_run_dictcheck_writes_known_unknown(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    wl = tmp_path / "wordlist.txt"
    wl.write_text("deus\n", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"g": {"files": ["a.txt"]}},
            "dictcheck": {"enabled": True, "wordlist": str(wl)},
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "expand_cleaned_dir_placeholders", lambda patterns, cleaned_dir: patterns)
    monkeypatch.setattr(runner_mod, "expand_globs", lambda patterns: [Path("a.txt")])
    monkeypatch.setattr(runner_mod, "read_concat", lambda files: "X")
    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: None)

    calls = []
    monkeypatch.setattr(runner_mod, "write_frequency_csv",
                        lambda path, c, header: calls.append(Path(path).name))

    rc = runner_mod.run(
        script_dir=script_dir,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=lambda *a, **k: Counter({"deus": 2, "angelus": 1}),
        render_stanza_package_table_fn=lambda *a, **k: [],
    )
    assert rc == 0

    assert calls == [
        "noun_frequency_g.csv",
        "noun_frequency_g.known.csv",
        "noun_frequency_g.unknown.csv",
    ]

def test_run_passes_filter_args_to_count_group(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    exc_file = tmp_path / "rom.txt"
    exc_file.write_text("vi\n", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "analysis_unit": "lemma",
            "groups": {"g": {"files": ["a.txt"]}},
            "filter": {
                "min_token_length": 3,
                "drop_roman_numerals": True,
                "roman_exceptions_file": "rom.txt"
            }
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "expand_cleaned_dir_placeholders", lambda patterns, cleaned_dir: patterns)
    monkeypatch.setattr(runner_mod, "expand_globs", lambda patterns: [Path("a.txt")])
    monkeypatch.setattr(runner_mod, "read_concat", lambda files: "X")

    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: None)
    monkeypatch.setattr(runner_mod, "write_frequency_csv", lambda *a, **k: None)

    captured_kwargs = {}
    def fake_count_group_fn(text, nlp, **kwargs):
        captured_kwargs.update(kwargs)
        return Counter({"X": 1})

    rc = runner_mod.run(
        script_dir=script_dir,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=fake_count_group_fn,
        render_stanza_package_table_fn=lambda *a, **k: [],
    )
    
    assert rc == 0
    assert captured_kwargs.get("min_token_length") == 3
    assert captured_kwargs.get("drop_roman_numerals") is True
    assert captured_kwargs.get("roman_exceptions_file") == (script_dir / "rom.txt").resolve()


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

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {"groups_files": kwargs["groups_files"]})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {})

    csv_calls = []
    monkeypatch.setattr(
        runner_mod,
        "write_frequency_csv",
        lambda path, c, header: csv_calls.append((Path(path).name, dict(c))),
    )

    meta_calls = []
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: meta_calls.append(meta))

    def count_group_fn(text, nlp, **kwargs):
        return Counter({text: 1})

    rc = runner_mod.run(
        project_root=project_root,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=count_group_fn,
        render_stanza_package_table_fn=lambda *a, **k: [],
    )

    assert rc == 0
    assert [name for name, _counter in csv_calls] == [
        "noun_frequency_file2.csv",
        "noun_frequency_virgil_aeneis.csv",
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

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {"groups_files": kwargs["groups_files"]})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: None)

    csv_names = []
    monkeypatch.setattr(
        runner_mod,
        "write_frequency_csv",
        lambda path, c, header: csv_names.append(Path(path).name),
    )

    rc = runner_mod.run(
        project_root=project_root,
        config_path=config_path,
        group_by_file=True,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=lambda *a, **k: Counter({"rosa": 1}),
        render_stanza_package_table_fn=lambda *a, **k: [],
    )

    assert rc == 0
    assert csv_names == ["noun_frequency_file3.csv"]


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

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)

    rc = runner_mod.run(
        project_root=project_root,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=lambda *a, **k: Counter({"rosa": 1}),
        render_stanza_package_table_fn=lambda *a, **k: [],
    )

    assert rc == 0
    meta = json.loads((project_root / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["groups_files"] == {"text": [str(used.resolve())]}
    generated_names = {Path(p).name for p in meta["generated_outputs"]}
    assert generated_names == {"noun_frequency_text.csv", "summary.txt", "run_meta.json"}


def test_run_error_on_empty_group(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {"out_dir": "output", "groups": {"empty": {"files": ["input/*.txt"]}}}

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)

    with pytest.raises(ValueError, match="No files matched"):
        runner_mod.run(
            project_root=project_root,
            config_path=config_path,
            load_config_fn=load_config_fn,
            clean_mod=object(),
            build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
            build_sentence_splitter_fn=None,
            count_group_fn=lambda *a, **k: Counter(),
            render_stanza_package_table_fn=lambda *a, **k: [],
            error_on_empty_group=True,
        )


def test_run_auto_single_cleaned_records_selected_file(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    selected = cleaned_dir / "satyricon.cleaned.txt"
    selected.write_text("rosa", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"old": {"files": ["cleaned/*.txt"]}},
            "grouping": {"mode": "auto_single_cleaned", "auto_group_name": "text"},
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: cleaned_dir)

    rc = runner_mod.run(
        project_root=project_root,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=lambda *a, **k: Counter({"rosa": 1}),
        render_stanza_package_table_fn=lambda *a, **k: [],
    )

    assert rc == 0
    meta = json.loads((project_root / "output" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["grouping"] == {"mode": "auto_single_cleaned", "auto_group_name": "text"}
    assert meta["groups_files"] == {"text": [str(selected.resolve())]}


def test_run_auto_single_cleaned_errors_on_zero_files(tmp_path: Path, monkeypatch):
    project_root = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"old": {"files": ["cleaned/*.txt"]}},
            "grouping": {"mode": "auto_single_cleaned"},
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: cleaned_dir)

    with pytest.raises(ValueError, match="no \\.txt files"):
        runner_mod.run(
            project_root=project_root,
            config_path=config_path,
            load_config_fn=load_config_fn,
            clean_mod=object(),
            build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
            build_sentence_splitter_fn=None,
            count_group_fn=lambda *a, **k: Counter(),
            render_stanza_package_table_fn=lambda *a, **k: [],
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

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"old": {"files": ["cleaned/*.txt"]}},
            "grouping": {"mode": "auto_single_cleaned"},
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: cleaned_dir)

    with pytest.raises(ValueError, match="expected exactly one"):
        runner_mod.run(
            project_root=project_root,
            config_path=config_path,
            load_config_fn=load_config_fn,
            clean_mod=object(),
            build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
            build_sentence_splitter_fn=None,
            count_group_fn=lambda *a, **k: Counter(),
            render_stanza_package_table_fn=lambda *a, **k: [],
        )
