from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.count_vocabula.dry_run import dry_run_count_vocabula


def test_dry_run_reports_config_paths_matches_and_warnings(tmp_path: Path, capsys):
    project_root = tmp_path
    (project_root / "config").mkdir()
    (project_root / "input").mkdir()
    (project_root / "cleaned").mkdir()
    (project_root / "data" / "wordlist").mkdir(parents=True)

    for name in ("a.txt", "b.txt", "c.txt"):
        (project_root / "input" / name).write_text("raw", encoding="utf-8")
        (project_root / "cleaned" / name).write_text("cleaned", encoding="utf-8")

    (project_root / "data" / "wordlist" / "latin_words.txt").write_text("rosa\n", encoding="utf-8")
    (project_root / "config" / "ref_tags.txt").write_text("ST\n", encoding="utf-8")
    (project_root / "config" / "roman_numeral_exceptions.txt").write_text("vi\n", encoding="utf-8")
    (project_root / "config" / "cleaner.yml").write_text(
        "\n".join(
            [
                "kind: scholastic_text",
                "input: ../input",
                "output: ../cleaned",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config_path = project_root / "config" / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "preprocess:",
                "  kind: cleaner",
                "  config: config/cleaner.yml",
                "groups:",
                "  text:",
                '    files: ["{cleaned_dir}/*.txt"]',
                "out_dir: output",
                "language: la",
                "stanza_package: perseus",
                "cpu_only: true",
                "trace:",
                "  enabled: true",
                "  max_rows: 20000",
                "trace:",
                "  enabled: true",
                "  max_rows: 500000",
                "filters:",
                "  min_token_length: 2",
                "  drop_roman_numerals: true",
                "  roman_exception_files: config/roman_numeral_exceptions.txt",
                "dictcheck:",
                "  enabled: true",
                "  wordlist: data/wordlist/latin_words.txt",
                "ref_tags:",
                "  enabled: true",
                "  patterns: config/ref_tags.txt",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = dry_run_count_vocabula(project_root=project_root, config_path=config_path)

    out = capsys.readouterr().out
    assert rc == 0
    assert "[OK] config loaded" in out
    assert "[OK] preprocess cleaner config found: config/cleaner.yml" in out
    assert "[OK] input files: 3" in out
    assert "[OK] cleaned output dir: cleaned" in out
    assert "[OK] group text matched files: 3" in out
    assert "  - cleaned/a.txt" in out
    assert "  - cleaned/b.txt" in out
    assert "  - cleaned/c.txt" in out
    assert "[WARN] duplicate YAML key: trace" in out
    assert "[WARN] unknown config key: roman_exception_files" in out
    assert "[OK] dictcheck wordlist found" in out
    assert "[OK] ref_tags patterns found" in out
    assert "[OK] output dir: output" in out


def test_dry_run_error_on_empty_group(tmp_path: Path, capsys):
    project_root = tmp_path
    (project_root / "config").mkdir()
    config_path = project_root / "config" / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  empty:",
                "    files:",
                "      - input/*.txt",
                "out_dir: output",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = dry_run_count_vocabula(
        project_root=project_root,
        config_path=config_path,
        error_on_empty_group=True,
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[ERROR] group empty matched files: 0" in out
