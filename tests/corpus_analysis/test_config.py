from __future__ import annotations

from pathlib import Path
import pytest

from nlpo_toolkit.corpus_analysis.config import AppConfig, config_to_dict, load_config


def test_load_config_accepts_preprocess_and_groups(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "preprocess:",
                "  kind: cleaner",
                "  config: cleaners/config/sample.yml",
                "groups:",
                "  text:",
                "    files:",
                "      - cleaned/*.txt",
                "out_dir: output",
                "language: la",
                "stanza_package: perseus",
                "cpu_only: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert isinstance(cfg, AppConfig)
    assert cfg.preprocess.kind == "cleaner"
    assert cfg.preprocess.config == "cleaners/config/sample.yml"
    assert "text" in cfg.groups
    assert cfg.groups["text"].files == ("cleaned/*.txt",)
    assert cfg.analysis_unit == "lemma"
    assert cfg.filters.min_token_length == 0
    assert cfg.filters.upos_targets == frozenset({"NOUN"})
    assert cfg.grouping.mode == "groups"
    assert cfg.trace.enabled is False
    assert cfg.dictcheck.wordlist is None
    assert cfg.nlp.backend == "stanza"


def test_load_config_normalizes_group_to_groups(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "group:",
                "  name: text",
                "  files:",
                "    - input/*.txt",
                "out_dir: output",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert "text" in cfg.groups
    assert cfg.groups["text"].files == ("input/*.txt",)


def test_load_config_rejects_missing_groups_and_group(tmp_path: Path):
    cfg_path = tmp_path / "invalid.yml"
    cfg_path.write_text("out_dir: output\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"define 'groups' or 'group'"):
        load_config(cfg_path)


def test_load_config_rejects_deprecated_cleaner_config(tmp_path: Path):
    cfg_path = tmp_path / "old.yml"
    cfg_path.write_text("cleaner_config: cleaners/config/sample.yml\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Deprecated key 'cleaner_config'"):
        load_config(cfg_path)


def test_load_config_accepts_grouping_per_file(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "groups:",
                "  text:",
                "    files:",
                "      - input/*.txt",
                "grouping:",
                "  mode: per_file",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.grouping.mode == "per_file"


def test_load_config_accepts_auto_single_cleaned_without_groups(tmp_path: Path):
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "grouping:",
                "  mode: auto_single_cleaned",
                "  auto_group_name: text",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.grouping.mode == "auto_single_cleaned"
    assert cfg.groups["text"].files == ()


def test_load_config_rejects_invalid_grouping_mode(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "groups:",
                "  text:",
                "    files:",
                "      - input/*.txt",
                "grouping:",
                "  mode: by_author",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"grouping\.mode"):
        load_config(cfg_path)


def test_load_config_nested_values_and_config_to_dict(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "groups:",
                "  corpus_a: {files: [input/corpus_a.txt]}",
                "nlp:",
                "  backend: transformers",
                "  language: xx",
                "  package: package_from_nlp",
                "  model_name: model_a",
                "language: la",
                "stanza_package: perseus",
                "cpu_only: true",
                "upos_targets: [NOUN, PROPN]",
                "filters:",
                "  min_token_length: 2",
                "  drop_roman_numerals: true",
                "  roman_exception_files: config/roman.txt",
                "trace:",
                "  enabled: true",
                "  max_rows: 10",
                "lemma_cache:",
                "  dir: cache/lemmas",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    data = config_to_dict(cfg)

    assert cfg.nlp.backend == "transformers"
    assert cfg.nlp.language == "la"
    assert cfg.nlp.stanza_package == "perseus"
    assert cfg.nlp.model_name == "model_a"
    assert cfg.filters.upos_targets == frozenset({"NOUN", "PROPN"})
    assert cfg.filters.roman_exceptions_file == "config/roman.txt"
    assert cfg.trace.max_rows == 10
    assert cfg.lemma_cache.directory == "cache/lemmas"
    assert data["groups"] == {"corpus_a": {"files": ["input/corpus_a.txt"]}}
    assert data["upos_targets"] == ["NOUN", "PROPN"]


def test_load_config_rejects_both_roman_exception_keys(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "groups:",
                "  corpus_a: {files: [input/corpus_a.txt]}",
                "filters:",
                "  roman_exceptions_file: config/roman_a.txt",
                "  roman_exception_files: config/roman_b.txt",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Specify only one"):
        load_config(cfg_path)


def test_load_config_uses_nlp_section_when_top_level_nlp_values_absent(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "groups:",
                "  corpus_a: {files: [input/corpus_a.txt]}",
                "nlp:",
                "  language: zz",
                "  package: package_a",
                "  cpu_only: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.nlp.language == "zz"
    assert cfg.nlp.stanza_package == "package_a"
    assert cfg.nlp.cpu_only is False


def test_load_config_rejects_top_level_non_mapping(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("- bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Top-level YAML"):
        load_config(cfg_path)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("groups:\n  corpus_a:\n    files: input/*.txt\n", "groups.corpus_a.files"),
        ("groups:\n  corpus_a:\n    files: [input/a.txt, 1]\n", "groups.corpus_a.files\\[1\\]"),
        ("groups:\n  corpus_a: {files: [input/a.txt]}\nanalysis_unit: token\n", "analysis_unit"),
        ("groups:\n  corpus_a: {files: [input/a.txt]}\nfilters:\n  min_token_length: -1\n", "filters.min_token_length"),
        ("groups:\n  corpus_a: {files: [input/a.txt]}\nfilters:\n  min_token_length: true\n", "filters.min_token_length"),
        ("groups:\n  corpus_a: {files: [input/a.txt]}\nupos_targets: NOUN\n", "upos_targets"),
        ("groups:\n  corpus_a: {files: [input/a.txt]}\ntrace:\n  max_rows: -1\n", "trace.max_rows"),
    ],
)
def test_load_config_rejects_invalid_typed_values(tmp_path: Path, body: str, match: str):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_config(cfg_path)


def test_load_config_loads_repository_config():
    cfg = load_config(Path("config/groups.config.yml"))

    assert isinstance(cfg, AppConfig)
    assert cfg.groups
