from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.corpus_analysis.dry_run import dry_run_count


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
                "nlp:",
                "  language: la",
                "  stanza_package: perseus",
                "  cpu_only: true",
                "trace:",
                "  enabled: true",
                "  max_rows: 20000",
                "trace:",
                "  enabled: true",
                "  max_rows: 500000",
                "filters:",
                "  min_token_length: 2",
                "  drop_roman_numerals: true",
                "  roman_exceptions_file: config/roman_numeral_exceptions.txt",
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

    rc = dry_run_count(project_root=project_root, config_path=config_path)

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

    rc = dry_run_count(
        project_root=project_root,
        config_path=config_path,
        error_on_empty_group=True,
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[ERROR] group empty matched files: 0" in out


def test_dry_run_auto_single_cleaned_reports_selected_file(tmp_path: Path, capsys):
    project_root = tmp_path
    (project_root / "config").mkdir()
    cleaned_dir = project_root / "cleaned"
    cleaned_dir.mkdir()
    selected = cleaned_dir / "satyricon.cleaned.txt"
    selected.write_text("cleaned", encoding="utf-8")
    (cleaned_dir / ".DS_Store").write_text("ignored", encoding="utf-8")
    (project_root / "config" / "cleaner.yml").write_text(
        "kind: scholastic_text\ninput: ../input\noutput: ../cleaned\n",
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
                "grouping:",
                "  mode: auto_single_cleaned",
                "  auto_group_name: text",
                "out_dir: output",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = dry_run_count(project_root=project_root, config_path=config_path)

    out = capsys.readouterr().out
    assert rc == 0
    assert "[OK] grouping mode: auto_single_cleaned" in out
    assert "[OK] auto selected cleaned file: cleaned/satyricon.cleaned.txt" in out
    assert "[OK] group text matched files: 1" in out
    assert "  - cleaned/satyricon.cleaned.txt" in out


def test_dry_run_reports_partition_validation_ok(tmp_path: Path, capsys):
    project_root = tmp_path
    (project_root / "config").mkdir()
    (project_root / "input").mkdir()
    for name in ("full", "part_a", "part_b"):
        (project_root / "input" / f"{name}.txt").write_text(name, encoding="utf-8")
    config_path = project_root / "config" / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  full: {files: [input/full.txt]}",
                "  part_a: {files: [input/part_a.txt]}",
                "  part_b: {files: [input/part_b.txt]}",
                "validations:",
                "  partitions:",
                "    - {name: full_split, whole: full, parts: [part_a, part_b]}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = dry_run_count(project_root=project_root, config_path=config_path)

    out = capsys.readouterr().out
    assert rc == 0
    assert "[OK] partition full_split: whole=full parts=part_a,part_b" in out


def test_dry_run_partition_empty_reference_is_error(tmp_path: Path, capsys):
    project_root = tmp_path
    (project_root / "config").mkdir()
    (project_root / "input").mkdir()
    (project_root / "input" / "full.txt").write_text("full", encoding="utf-8")
    (project_root / "input" / "part_a.txt").write_text("part_a", encoding="utf-8")
    config_path = project_root / "config" / "groups.config.yml"
    config_path.write_text(
        "\n".join(
            [
                "groups:",
                "  full: {files: [input/full.txt]}",
                "  part_a: {files: [input/part_a.txt]}",
                "  part_b: {files: [input/missing.txt]}",
                "validations:",
                "  partitions:",
                "    - {name: full_split, whole: full, parts: [part_a, part_b]}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rc = dry_run_count(project_root=project_root, config_path=config_path)

    out = capsys.readouterr().out
    assert rc == 1
    assert "[ERROR] partition full_split references empty group: part_b" in out
