from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.count_command import CountRequest
from nlpo_toolkit.corpus_analysis.dependencies import CorpusPlanningDependencies
from nlpo_toolkit.corpus_analysis.dry_run import execute_dry_run
from nlpo_toolkit.corpus_analysis.config import load_config
from nlpo_toolkit.corpus_analysis.config import ConfigError


def _execute_dry_run(
    *,
    project_root: Path,
    config_path: Path,
    group_by_file: bool = False,
    error_on_empty_group: bool = False,
    auto_single_cleaned: bool = False,
) -> int:
    return execute_dry_run(
        request=CountRequest(
            project_root=project_root,
            config_path=config_path,
            group_by_file=group_by_file,
            error_on_empty_group=error_on_empty_group,
            auto_single_cleaned=auto_single_cleaned,
            dry_run=True,
        ),
        dependencies=CorpusPlanningDependencies(
            load_config=load_config,
            cleaner_loader=lambda: pytest.fail(
                "cleaner loader must not be called"
            ),
        ),
    )


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

    rc = _execute_dry_run(project_root=project_root, config_path=config_path)

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

    rc = _execute_dry_run(
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
        "kind: scholastic_text\ninput: ../cleaned\noutput: ../cleaned\n",
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

    rc = _execute_dry_run(project_root=project_root, config_path=config_path)

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

    rc = _execute_dry_run(project_root=project_root, config_path=config_path)

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

    rc = _execute_dry_run(project_root=project_root, config_path=config_path)

    out = capsys.readouterr().out
    assert rc == 1
    assert "[ERROR] partition full_split references empty group: part_b" in out


@pytest.mark.parametrize(
    ("contents", "binary"),
    (
        ("groups: [\n", False),
        (b"\xff\xfe", True),
        ("- not\n- a\n- mapping\n", False),
        ("groups: {}\nunknown_key: true\n", False),
    ),
)
def test_dry_run_reports_user_correctable_config_errors(
    tmp_path: Path,
    capsys,
    contents: str | bytes,
    binary: bool,
) -> None:
    config_path = tmp_path / "broken.yml"
    if binary:
        assert isinstance(contents, bytes)
        config_path.write_bytes(contents)
    else:
        assert isinstance(contents, str)
        config_path.write_text(contents, encoding="utf-8")

    rc = _execute_dry_run(project_root=tmp_path, config_path=config_path)

    assert rc == 1
    assert "[ERROR] config:" in capsys.readouterr().out


def test_dry_run_reports_missing_config_and_preserves_cause(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "missing.yml"

    rc = _execute_dry_run(project_root=tmp_path, config_path=config_path)

    assert rc == 1
    assert "missing.yml" in capsys.readouterr().out
    with pytest.raises(ConfigError) as caught:
        load_config(config_path)
    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize(
    "error",
    (
        RuntimeError("programmer bug"),
        TypeError("wrong internal call"),
        AssertionError("unexpected state"),
    ),
)
def test_dry_run_does_not_hide_programmer_errors(
    tmp_path: Path,
    error: Exception,
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("groups: {}\n", encoding="utf-8")

    def broken_loader(_path: Path):
        raise error

    with pytest.raises(type(error), match=str(error)):
        execute_dry_run(
            request=CountRequest(
                project_root=tmp_path,
                config_path=config_path,
                dry_run=True,
            ),
            dependencies=CorpusPlanningDependencies(
                load_config=broken_loader,
                cleaner_loader=lambda: pytest.fail(
                    "cleaner loader must not be called"
                ),
            ),
        )


@pytest.mark.parametrize(
    "cleaner_contents",
    (
        "groups: [\n",
        "- not-a-mapping\n",
        "input: ../input\n",
    ),
)
def test_dry_run_reports_invalid_cleaner_config(
    tmp_path: Path,
    capsys,
    cleaner_contents: str,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cleaner.yml").write_text(cleaner_contents, encoding="utf-8")
    config_path = config_dir / "groups.yml"
    config_path.write_text(
        "preprocess:\n"
        "  kind: cleaner\n"
        "  config: config/cleaner.yml\n"
        "groups:\n"
        "  text: {files: ['{cleaned_dir}/*.txt']}\n",
        encoding="utf-8",
    )

    rc = _execute_dry_run(project_root=tmp_path, config_path=config_path)

    assert rc == 1
    assert "[ERROR]" in capsys.readouterr().out


@pytest.mark.parametrize("create_invalid_utf8", (False, True))
def test_dry_run_reports_missing_or_non_utf8_cleaner_config(
    tmp_path: Path,
    capsys,
    create_invalid_utf8: bool,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    if create_invalid_utf8:
        (config_dir / "cleaner.yml").write_bytes(b"\xff\xfe")
    config_path = config_dir / "groups.yml"
    config_path.write_text(
        "preprocess:\n"
        "  kind: cleaner\n"
        "  config: config/cleaner.yml\n"
        "groups:\n"
        "  text: {files: ['{cleaned_dir}/*.txt']}\n",
        encoding="utf-8",
    )

    rc = _execute_dry_run(project_root=tmp_path, config_path=config_path)

    assert rc == 1
    output = capsys.readouterr().out
    assert "[ERROR]" in output
    assert "cleaner.yml" in output
