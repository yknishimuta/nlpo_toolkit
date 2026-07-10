from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pytest
import yaml

from nlpo_toolkit.corpus_analysis.config import (
    AnalysisCacheConfig,
    AppConfig,
    LemmaCacheConfig,
    config_to_dict,
    ensure_app_config,
    load_config,
)


def _assert_cache_sections_exclusive(serialized: dict[str, object]) -> None:
    assert not (
        "analysis_cache" in serialized
        and "lemma_cache" in serialized
    )


def _assert_config_round_trip(config: AppConfig) -> dict[str, object]:
    serialized = config_to_dict(config)
    _assert_cache_sections_exclusive(serialized)
    assert ensure_app_config(serialized) == config
    return serialized


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
    assert cfg.analysis_cache.directory == "cache/lemmas"
    assert data["groups"] == {"corpus_a": {"files": ["input/corpus_a.txt"]}}
    assert data["upos_targets"] == ["NOUN", "PROPN"]
    assert "lemma_cache" in data
    assert "analysis_cache" not in data
    assert ensure_app_config(data) == cfg


def test_load_config_parses_analysis_cache_and_rejects_legacy_conflict(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "groups:",
                "  corpus_a: {files: [input/corpus_a.txt]}",
                "analysis_cache:",
                "  enabled: true",
                "  dir: .analysis_cache",
                "  manifest_key_mode: relative",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.analysis_cache.enabled is True
    assert cfg.analysis_cache.directory == ".analysis_cache"
    data = config_to_dict(cfg)
    _assert_cache_sections_exclusive(data)
    assert "analysis_cache" in data
    assert "lemma_cache" not in data
    assert ensure_app_config(data) == cfg

    cfg_path.write_text(
        "\n".join(
            [
                "groups:",
                "  corpus_a: {files: [input/corpus_a.txt]}",
                "analysis_cache: {enabled: true}",
                "lemma_cache: {enabled: true}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Specify only one of analysis_cache"):
        load_config(cfg_path)


def test_config_to_dict_round_trips_minimal_config():
    original = ensure_app_config(
        {
            "groups": {
                "text": {
                    "files": ["input/*.txt"],
                }
            }
        }
    )

    serialized = _assert_config_round_trip(original)

    assert "preprocess" not in serialized


def test_config_to_dict_round_trips_cleaner_and_full_sections():
    original = ensure_app_config(
        {
            "groups": {
                "text": {"files": ["input/a.txt", "input/b.txt"]},
                "other": {"files": ["input/other.txt"]},
            },
            "preprocess": {
                "kind": "cleaner",
                "config": "cleaners/config/sample.yml",
            },
            "grouping": {
                "mode": "groups",
                "auto_group_name": "auto_text",
            },
            "nlp": {
                "backend": "stanza",
                "language": "la",
                "stanza_package": "perseus",
                "cpu_only": False,
            },
            "filters": {
                "min_token_length": 2,
                "drop_roman_numerals": True,
                "roman_exceptions_file": "config/roman.txt",
                "exclude_lemmas": "config/exclude.txt",
            },
            "upos_targets": ["PROPN", "NOUN"],
            "normalization": {
                "enabled": True,
                "casefold": True,
                "uv": "v",
                "ij": "i",
                "diacritics": "strip",
                "ligatures": {"æ": "ae"},
                "map_u_v": True,
                "map_i_j": True,
                "strip_diacritics": True,
                "normalize_ligatures": True,
                "unicode_nf": "NFC",
            },
            "dictcheck": {
                "enabled": True,
                "wordlist": "dict/latin.txt",
                "lemma_normalize": "dict/normalize.yml",
            },
            "ref_tags": {
                "enabled": True,
                "patterns": "config/ref_tags.yml",
            },
            "trace": {
                "enabled": True,
                "path": "output/trace.tsv",
                "max_rows": 100,
                "only_keys": ["arma", "vir"],
                "write_truncation_marker": False,
            },
            "artifacts": {
                "tokens": {
                    "enabled": True,
                    "path": "output/tokens.tsv",
                }
            },
            "archive": {
                "enabled": True,
                "runs_dir": "runs",
                "include_input": True,
                "include_cleaned": True,
            },
            "analysis_cache": {
                "enabled": True,
                "dir": ".analysis_cache",
                "use_manifest": False,
                "manifest_key_mode": "relative",
                "lock_timeout_sec": 45.5,
            },
            "prune": {
                "keep_days": 7,
                "keep_files": 20,
                "lock_ttl_sec": 60,
            },
            "analysis_unit": "surface",
            "csv_header": ["form", "count"],
            "vocab_path": "config/vocab.tsv",
            "out_dir": "custom-output",
        }
    )

    serialized = _assert_config_round_trip(original)

    assert serialized["preprocess"] == {
        "kind": "cleaner",
        "config": "cleaners/config/sample.yml",
    }
    assert "analysis_cache" in serialized
    assert "lemma_cache" not in serialized


def test_config_to_dict_round_trips_partition_validation():
    original = ensure_app_config(
        {
            "groups": {
                "whole": {"files": ["input/whole.txt"]},
                "part_a": {"files": ["input/a.txt"]},
                "part_b": {"files": ["input/b.txt"]},
            },
            "validations": {
                "partitions": [
                    {
                        "name": "split",
                        "whole": "whole",
                        "parts": ["part_a", "part_b"],
                        "on_mismatch": "error",
                        "report": "all",
                    }
                ]
            },
        }
    )

    serialized = _assert_config_round_trip(original)
    validations = serialized["validations"]
    assert isinstance(validations, dict)
    partitions = validations["partitions"]
    assert isinstance(partitions, list)
    assert isinstance(partitions[0]["parts"], list)
    restored = ensure_app_config(serialized)
    assert restored.partition_validations == original.partition_validations


def test_config_to_dict_round_trips_comparisons():
    original = ensure_app_config(
        {
            "groups": {
                "a": {"files": ["input/a.txt"]},
                "b": {"files": ["input/b.txt"]},
            },
            "comparisons": [
                {
                    "name": "a_vs_b",
                    "group_a": "a",
                    "group_b": "b",
                    "scale": 1000,
                    "zero_correction": 0.25,
                    "min_total_count": 3,
                    "report": "filtered",
                    "sort": {
                        "by": "abs_log_ratio",
                        "descending": False,
                    },
                }
            ],
        }
    )

    serialized = _assert_config_round_trip(original)
    assert serialized["comparisons"] == [
        {
            "name": "a_vs_b",
            "group_a": "a",
            "group_b": "b",
            "scale": 1000,
            "zero_correction": 0.25,
            "min_total_count": 3,
            "report": "filtered",
            "sort": {
                "by": "abs_log_ratio",
                "descending": False,
            },
        }
    ]


def test_config_to_dict_round_trips_legacy_lemma_cache():
    original = ensure_app_config(
        {
            "groups": {
                "text": {"files": ["input/*.txt"]},
            },
            "lemma_cache": {
                "enabled": True,
                "dir": "cache/lemmas",
                "use_manifest": False,
                "lock_timeout_sec": 12.5,
                "include_ref_tags_in_config_hash": False,
            },
        }
    )

    serialized = _assert_config_round_trip(original)

    assert "lemma_cache" in serialized
    assert "analysis_cache" not in serialized


def test_config_to_dict_rejects_incompatible_cache_sections():
    base = ensure_app_config(
        {
            "groups": {
                "text": {"files": ["input/*.txt"]},
            },
        }
    )
    incompatible = replace(
        base,
        analysis_cache=AnalysisCacheConfig(
            enabled=True,
            directory="analysis-cache",
        ),
        lemma_cache=LemmaCacheConfig(
            enabled=True,
            directory="lemma-cache",
        ),
    )

    with pytest.raises(ValueError, match="incompatible analysis_cache and lemma_cache"):
        config_to_dict(incompatible)


def test_config_to_dict_output_is_yaml_safe():
    original = ensure_app_config(
        {
            "groups": {
                "text": {"files": ["input/*.txt"]},
            },
            "analysis_cache": {
                "enabled": True,
                "dir": ".analysis_cache",
            },
            "validations": {
                "partitions": [],
            },
        }
    )

    serialized = config_to_dict(original)
    yaml_text = yaml.safe_dump(serialized, sort_keys=False)
    loaded = yaml.safe_load(yaml_text)
    restored = ensure_app_config(loaded)

    _assert_cache_sections_exclusive(serialized)
    assert restored == original


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
