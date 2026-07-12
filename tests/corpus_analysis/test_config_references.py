from pathlib import Path

from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.config_references import (
    build_config_file_inventory,
)
from nlpo_toolkit.latin.cleaners.config_loader import inspect_cleaner_config


def test_inventory_uses_typed_cleaner_inspection(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("a", encoding="utf-8")
    root_config = config_dir / "groups.yml"
    root_config.write_text("groups: {}\n", encoding="utf-8")
    rules = config_dir / "rules.yml"
    lexicon = config_dir / "lexicon.tsv"
    rules.write_text("rules: []\n", encoding="utf-8")
    lexicon.write_text("a\tb\n", encoding="utf-8")
    cleaner = config_dir / "cleaner.yml"
    cleaner.write_text(
        "kind: corpus_corporum\n"
        "input: ../input\n"
        "output: ../cleaned\n"
        "rules_path: rules.yml\n"
        "lexicon_map_path: lexicon.tsv\n",
        encoding="utf-8",
    )
    config = ensure_app_config({"groups": {"text": {"files": []}}})

    inventory = build_config_file_inventory(
        config=config,
        config_path=root_config,
        project_root=tmp_path,
        cleaner_inspection=inspect_cleaner_config(cleaner),
    )

    assert [(item.kind, item.path) for item in inventory] == [
        ("root_config", root_config.resolve()),
        ("preprocess.config", cleaner.resolve()),
        ("preprocess.rules_path", rules.resolve()),
        ("preprocess.lexicon_map_path", lexicon.resolve()),
    ]


def test_inventory_without_cleaner_has_no_preprocess_entries(tmp_path: Path) -> None:
    root_config = tmp_path / "groups.yml"
    root_config.write_text("groups: {}\n", encoding="utf-8")
    inventory = build_config_file_inventory(
        config=ensure_app_config({"groups": {"text": {"files": []}}}),
        config_path=root_config,
        project_root=tmp_path,
        cleaner_inspection=None,
    )
    assert [item.kind for item in inventory] == ["root_config"]
