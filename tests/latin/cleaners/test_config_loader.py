from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.cleaner_contracts import (
    CleanerConfigParseError,
    CleanerConfigReadError,
    CleanerConfigValidationError,
)
from nlpo_toolkit.latin.cleaners.config_loader import (
    inspect_cleaner_config,
    load_cleaner_config,
)


def test_load_cleaner_config_resolves_typed_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("a.txt", "b.TXT", "note.md"):
        (input_dir / name).write_text(name, encoding="utf-8")
    nested = input_dir / "nested"
    nested.mkdir()
    (nested / "c.txt").write_text("nested", encoding="utf-8")
    rules = config_dir / "rules.yml"
    lexicon = config_dir / "lexicon.tsv"
    rules.write_text("rules: []\n", encoding="utf-8")
    lexicon.write_text("a\tb\n", encoding="utf-8")
    path = config_dir / "cleaner.yml"
    path.write_text(
        "kind: corpus_corporum\n"
        "input: ../input\n"
        "output: ../cleaned\n"
        "rules_path: rules.yml\n"
        "lexicon_map_path: lexicon.tsv\n",
        encoding="utf-8",
    )

    config = load_cleaner_config(path)
    inspection = inspect_cleaner_config(path)

    assert config.source_path == path.resolve()
    assert config.input_path == input_dir.resolve()
    assert config.output_path == (tmp_path / "cleaned").resolve()
    assert config.rules_path == rules.resolve()
    assert config.lexicon_map_path == lexicon.resolve()
    assert inspection.input_files == (
        (input_dir / "a.txt").resolve(),
        (input_dir / "b.TXT").resolve(),
    )
    assert [(item.kind, item.path) for item in inspection.referenced_files] == [
        ("preprocess.rules_path", rules.resolve()),
        ("preprocess.lexicon_map_path", lexicon.resolve()),
    ]


@pytest.mark.parametrize(
    "contents",
    (
        "input: input\noutput: output\n",
        "kind: x\noutput: output\n",
        "kind: x\ninput: input\n",
        "kind: unknown\ninput: input\noutput: output\n",
        "- invalid-root\n",
    ),
)
def test_load_cleaner_config_rejects_invalid_schema(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "cleaner.yml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(CleanerConfigValidationError):
        load_cleaner_config(path)


def test_load_cleaner_config_reports_read_and_yaml_errors(tmp_path: Path) -> None:
    with pytest.raises(CleanerConfigReadError) as missing:
        load_cleaner_config(tmp_path / "missing.yml")
    assert "Failed to read YAML file" in str(missing.value)

    invalid_utf8 = tmp_path / "utf8.yml"
    invalid_utf8.write_bytes(b"\xff\xfe")
    with pytest.raises(CleanerConfigReadError) as decoded:
        load_cleaner_config(invalid_utf8)
    assert "not valid UTF-8" in str(decoded.value)

    invalid_yaml = tmp_path / "yaml.yml"
    invalid_yaml.write_text("kind: [\n", encoding="utf-8")
    with pytest.raises(CleanerConfigParseError):
        load_cleaner_config(invalid_yaml)


@pytest.mark.parametrize("contents", (
    "kind: corpus_corporum\nkind: scholastic_text\ninput: in\noutput: out\n",
    "kind: corpus_corporum\ninput: in\noutput: out\nnested:\n  key: a\n  key: b\n",
))
def test_cleaner_config_rejects_duplicate_keys(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "cleaner.yml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(CleanerConfigParseError) as caught:
        load_cleaner_config(path)
    assert "Duplicate YAML key" in str(caught.value)
    assert str(path.resolve()) in str(caught.value)
