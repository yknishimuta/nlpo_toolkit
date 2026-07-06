from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nlpo_toolkit.count_vocabula import cli
from nlpo_toolkit.count_vocabula.archive import (
    RunArchiveError,
    create_run_archive,
    file_sha256,
    sanitize_run_name,
)


def _write_project(tmp_path: Path) -> tuple[Path, Path, dict]:
    project_root = tmp_path
    (project_root / "config" / "latin_cleaners").mkdir(parents=True)
    (project_root / "input").mkdir()
    (project_root / "cleaned").mkdir()
    (project_root / "output").mkdir()
    (project_root / "data" / "wordlist").mkdir(parents=True)

    (project_root / "input" / "a.txt").write_text("raw text\n", encoding="utf-8")
    (project_root / "cleaned" / "a.cleaned.txt").write_text("cleaned text\n", encoding="utf-8")
    (project_root / "output" / "noun_frequency_text.csv").write_text(
        "lemma,count\nrosa,1\n",
        encoding="utf-8",
    )
    (project_root / "output" / "noun_frequency_text.known.csv").write_text(
        "lemma,count\nrosa,1\n",
        encoding="utf-8",
    )
    (project_root / "output" / "summary.txt").write_text("# Summary\n", encoding="utf-8")
    (project_root / "output" / "run_meta.json").write_text("{}\n", encoding="utf-8")
    (project_root / "output" / "trace.tsv").write_text("token\tlemma\nrosa\trosa\n", encoding="utf-8")

    (project_root / "config" / "lemma_normalize.tsv").write_text("rosae\trosa\n", encoding="utf-8")
    (project_root / "config" / "ref_tags.txt").write_text("REF\n", encoding="utf-8")
    (project_root / "config" / "roman_numeral_exceptions.txt").write_text("vi\n", encoding="utf-8")
    (project_root / "config" / "exclude_lemmas.txt").write_text("sum\n", encoding="utf-8")
    (project_root / "config" / "latin_cleaners" / "subst_patterns.yml").write_text(
        "substitute_patterns: []\n",
        encoding="utf-8",
    )
    (project_root / "config" / "latin_cleaners" / "lexicon_map.tsv").write_text(
        "uod\tquod\n",
        encoding="utf-8",
    )
    (project_root / "data" / "wordlist" / "latin_words.txt").write_text("rosa\n", encoding="utf-8")
    (project_root / "config" / "cleaner.yml").write_text(
        "\n".join(
            [
                "input: ../input",
                "output: ../cleaned",
                "rules_path: latin_cleaners/subst_patterns.yml",
                "lexicon_map_path: latin_cleaners/lexicon_map.tsv",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
        "groups": {"text": {"files": ["input/*.txt"]}},
        "out_dir": "output",
        "dictcheck": {
            "enabled": True,
            "wordlist": "data/wordlist/latin_words.txt",
            "lemma_normalize": "config/lemma_normalize.tsv",
        },
        "ref_tags": {"enabled": True, "patterns": "config/ref_tags.txt"},
        "trace": {"enabled": True, "path": "output/trace.tsv"},
        "filters": {
            "roman_exceptions_file": "config/roman_numeral_exceptions.txt",
            "exclude_lemmas": "config/exclude_lemmas.txt",
        },
    }
    config_path = project_root / "config" / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "preprocess:",
                "  kind: cleaner",
                "  config: config/cleaner.yml",
                "groups:",
                "  text:",
                "    files:",
                "      - input/*.txt",
                "out_dir: output",
                "dictcheck:",
                "  enabled: true",
                "  wordlist: data/wordlist/latin_words.txt",
                "  lemma_normalize: config/lemma_normalize.tsv",
                "ref_tags:",
                "  enabled: true",
                "  patterns: config/ref_tags.txt",
                "trace:",
                "  enabled: true",
                "  path: output/trace.tsv",
                "filters:",
                "  roman_exceptions_file: config/roman_numeral_exceptions.txt",
                "  exclude_lemmas: config/exclude_lemmas.txt",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return project_root, config_path, config


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("virgil noun test 01", "virgil_noun_test_01"),
        ("Virgil: noun/test?", None),
        ("  a---b  ", "a---b"),
    ],
)
def test_sanitize_run_name(raw: str, expected: str | None) -> None:
    if expected is None:
        with pytest.raises(ValueError):
            sanitize_run_name(raw)
    else:
        assert sanitize_run_name(raw) == expected


@pytest.mark.parametrize("bad", ["", "../x", "/tmp/x", "a/../b"])
def test_sanitize_run_name_rejects_unsafe_names(bad: str) -> None:
    with pytest.raises(ValueError):
        sanitize_run_name(bad)


def test_create_run_archive_with_run_name_creates_expected_files(tmp_path: Path) -> None:
    project_root, config_path, config = _write_project(tmp_path)

    run_dir = create_run_archive(
        project_root=project_root,
        config_path=config_path,
        config=config,
        run_name="virgil noun test 01",
        command_line=["nlpo", "count-vocabula", "--run-name", "virgil noun test 01"],
    )

    assert run_dir == project_root / "runs" / "virgil_noun_test_01"
    assert (run_dir / "outputs" / "noun_frequency_text.csv").exists()
    assert (run_dir / "outputs" / "noun_frequency_text.known.csv").exists()
    assert (run_dir / "outputs" / "summary.txt").exists()
    assert (run_dir / "outputs" / "run_meta.json").exists()
    assert (run_dir / "trace" / "trace.tsv").exists()
    assert (run_dir / "config_snapshot" / "config" / "groups.config.yml").exists()
    assert (run_dir / "config_snapshot" / "config" / "cleaner.yml").exists()
    assert (run_dir / "config_snapshot" / "config" / "lemma_normalize.tsv").exists()
    assert (run_dir / "config_snapshot" / "config" / "ref_tags.txt").exists()
    assert (run_dir / "config_snapshot" / "config" / "roman_numeral_exceptions.txt").exists()
    assert (run_dir / "config_snapshot" / "config" / "latin_cleaners" / "subst_patterns.yml").exists()
    assert (run_dir / "config_snapshot" / "config" / "latin_cleaners" / "lexicon_map.tsv").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "README.md").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_name"] == "virgil_noun_test_01"
    assert manifest["input_files"][0]["sha256"] == file_sha256(project_root / "input" / "a.txt")
    assert manifest["input_files"][0]["size"] > 0
    assert manifest["output_files"][0]["sha256"]
    assert manifest["config_snapshot_files"][0]["sha256"]
    wordlist_ref = manifest["external_references"][0]
    assert wordlist_ref["kind"] == "dictcheck.wordlist"
    assert wordlist_ref["sha256"] == file_sha256(project_root / "data" / "wordlist" / "latin_words.txt")
    assert not (run_dir / "config_snapshot" / "data" / "wordlist" / "latin_words.txt").exists()


def test_create_run_archive_uses_timestamp_when_run_name_missing(tmp_path: Path) -> None:
    project_root, config_path, config = _write_project(tmp_path)

    run_dir = create_run_archive(
        project_root=project_root,
        config_path=config_path,
        config=config,
        run_name=None,
        created_at=datetime(2026, 7, 6, 12, 34, 56, tzinfo=timezone.utc),
    )

    assert run_dir.name == "20260706-123456"
    assert run_dir.exists()


def test_create_run_archive_does_not_overwrite_existing_dir(tmp_path: Path) -> None:
    project_root, config_path, config = _write_project(tmp_path)
    existing = project_root / "runs" / "existing"
    existing.mkdir(parents=True)

    with pytest.raises(RunArchiveError, match="already exists"):
        create_run_archive(
            project_root=project_root,
            config_path=config_path,
            config=config,
            run_name="existing",
        )

    assert list(existing.iterdir()) == []


def test_create_run_archive_copies_cleaned_and_input_only_when_requested(tmp_path: Path) -> None:
    project_root, config_path, config = _write_project(tmp_path)

    without_files = create_run_archive(
        project_root=project_root,
        config_path=config_path,
        config=config,
        run_name="without-files",
    )
    assert not (without_files / "cleaned").exists()
    assert not (without_files / "input").exists()

    with_files = create_run_archive(
        project_root=project_root,
        config_path=config_path,
        config=config,
        run_name="with-files",
        include_cleaned=True,
        include_input=True,
    )
    assert (with_files / "cleaned" / "a.cleaned.txt").exists()
    assert (with_files / "input" / "input" / "a.txt").exists()


def test_run_count_vocabula_creates_archive_after_success(tmp_path: Path, monkeypatch) -> None:
    project_root, config_path, _config = _write_project(tmp_path)

    def fake_run(**_kwargs) -> int:
        return 0

    monkeypatch.setattr(cli, "run", fake_run)

    rc = cli.run_count_vocabula(
        project_root=project_root,
        config_path=config_path,
        group_by_file=False,
        run_name="from-cli",
        command_line=["nlpo", "count-vocabula", "--run-name", "from-cli"],
    )

    assert rc == 0
    assert (project_root / "runs" / "from-cli" / "manifest.json").exists()
