from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nlpo_toolkit.corpus_analysis.config import (
    AppConfig,
    NormalizationConfig,
    config_to_dict,
    ensure_app_config,
    load_config,
)


def _assert_config_round_trip(config: AppConfig) -> dict[str, object]:
    serialized = config_to_dict(config)
    assert ensure_app_config(serialized) == config
    return serialized


def test_minimal_canonical_config():
    config = ensure_app_config(
        {
            "groups": {
                "text": {
                    "files": ["input/*.txt"],
                }
            }
        }
    )

    assert config.groups["text"].files == ("input/*.txt",)
    assert config.nlp.language == "la"
    assert config.filters.upos_targets == frozenset({"NOUN"})


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
                "nlp:",
                "  language: la",
                "  stanza_package: perseus",
                "  cpu_only: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert isinstance(cfg, AppConfig)
    assert cfg.preprocess.kind == "cleaner"
    assert cfg.preprocess.config == "cleaners/config/sample.yml"
    assert cfg.groups["text"].files == ("cleaned/*.txt",)
    assert cfg.analysis_unit == "lemma"
    assert cfg.filters.min_token_length == 0
    assert cfg.filters.upos_targets == frozenset({"NOUN"})
    assert cfg.grouping.mode == "groups"
    assert cfg.trace.enabled is False
    assert cfg.dictcheck.wordlist is None
    assert cfg.nlp.backend == "stanza"


def test_load_config_rejects_missing_groups(tmp_path: Path):
    cfg_path = tmp_path / "invalid.yml"
    cfg_path.write_text("out_dir: output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="groups: Field required"):
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


def test_nlp_settings_are_nested():
    config = ensure_app_config(
        {
            "groups": {
                "text": {
                    "files": ["input/*.txt"],
                }
            },
            "nlp": {
                "backend": "stanza",
                "language": "la",
                "stanza_package": "perseus",
                "cpu_only": False,
            },
        }
    )

    assert config.nlp.cpu_only is False


def test_filter_settings_are_nested():
    config = ensure_app_config(
        {
            "groups": {
                "text": {
                    "files": ["input/*.txt"],
                }
            },
            "filters": {
                "upos_targets": ["NOUN", "PROPN"],
                "min_token_length": 2,
                "drop_roman_numerals": True,
                "roman_exceptions_file": "config/roman.txt",
            },
        }
    )

    assert config.filters.upos_targets == frozenset({"NOUN", "PROPN"})
    assert config.filters.min_token_length == 2
    assert config.filters.drop_roman_numerals is True
    assert config.filters.roman_exceptions_file == "config/roman.txt"


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
                "  stanza_package: package_from_nlp",
                "  model_name: model_a",
                "  cpu_only: true",
                "filters:",
                "  upos_targets: [NOUN, PROPN]",
                "  min_token_length: 2",
                "  drop_roman_numerals: true",
                "  roman_exceptions_file: config/roman.txt",
                "trace:",
                "  enabled: true",
                "  max_rows: 10",
                "analysis_cache:",
                "  dir: cache/analysis",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    data = config_to_dict(cfg)

    assert cfg.nlp.backend == "transformers"
    assert cfg.nlp.language == "xx"
    assert cfg.nlp.stanza_package == "package_from_nlp"
    assert cfg.nlp.model_name == "model_a"
    assert cfg.filters.upos_targets == frozenset({"NOUN", "PROPN"})
    assert cfg.filters.roman_exceptions_file == "config/roman.txt"
    assert cfg.trace.max_rows == 10
    assert cfg.analysis_cache.directory == "cache/analysis"
    assert data["groups"] == {"corpus_a": {"files": ["input/corpus_a.txt"]}}
    assert data["filters"]["upos_targets"] == ["NOUN", "PROPN"]
    assert data["analysis_cache"]["dir"] == "cache/analysis"
    assert ensure_app_config(data) == cfg


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
                "upos_targets": ["PROPN", "NOUN"],
                "min_token_length": 2,
                "drop_roman_numerals": True,
                "roman_exceptions_file": "config/roman.txt",
            },
            "normalization": {
                "enabled": True,
                "casefold": True,
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
                "lock_timeout_sec": 45.5,
            },
            "analysis_unit": "surface",
            "csv_header": ["form", "count"],
            "out_dir": "custom-output",
        }
    )

    serialized = _assert_config_round_trip(original)

    assert serialized["preprocess"] == {
        "kind": "cleaner",
        "config": "cleaners/config/sample.yml",
    }
    assert "analysis_cache" in serialized
    assert ("lemma" + "_cache") not in serialized


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
    assert restored.validations.partitions == original.validations.partitions


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
            "sort": {
                "by": "abs_log_ratio",
                "descending": False,
            },
        }
    ]


def test_config_to_dict_round_trips_analysis_cache():
    original = ensure_app_config(
        {
            "groups": {
                "text": {"files": ["input/*.txt"]},
            },
            "analysis_cache": {
                "enabled": True,
                "dir": "cache/analysis",
                "lock_timeout_sec": 12.5,
            },
        }
    )

    serialized = _assert_config_round_trip(original)

    assert serialized["analysis_cache"] == {
        "enabled": True,
        "dir": "cache/analysis",
        "lock_timeout_sec": 12.5,
    }


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

    assert ("lemma" + "_cache") not in serialized
    assert restored == original


@pytest.mark.parametrize(
    "removed_key",
    [
        "cleaner_config",
        "cpu_only",
        "filter",
        "group",
        "language",
        "stanza_package",
        "stanza_pkg",
        "upos_targets",
        "vocab_path",
        "prune",
    ],
)
def test_removed_top_level_keys_are_rejected(removed_key: str):
    raw = {
        "groups": {
            "text": {
                "files": ["input/*.txt"],
            }
        },
        removed_key: True,
    }

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ensure_app_config(raw)


@pytest.mark.parametrize(
    ("section", "removed_key", "value"),
    [
        ("filters", "roman_exception_files", "config/roman.txt"),
        ("filters", "exclude_lemmas", "config/exclude.txt"),
        ("filters", "exclude_lemmas_file", "config/exclude.txt"),
        ("ref_tags", "ref_tags_file", "config/ref_tags.txt"),
        ("normalization", "uv", "v"),
        ("normalization", "ij", "i"),
        ("normalization", "diacritics", "strip"),
        ("normalization", "ligatures", {"æ": "ae"}),
    ],
)
def test_removed_nested_keys_are_rejected(section: str, removed_key: str, value: object):
    raw = {
        "groups": {
            "text": {
                "files": ["input/*.txt"],
            }
        },
        section: {
            removed_key: value,
        },
    }

    with pytest.raises(ValueError, match=f"{section}.{removed_key}.*Extra inputs"):
        ensure_app_config(raw)


def test_app_config_has_no_unused_vocab_path():
    names = set(AppConfig.model_fields)

    assert "vocab_path" not in names
    assert "prune" not in names


def test_normalization_config_has_only_active_fields():
    names = set(NormalizationConfig.model_fields)

    assert names == {
        "enabled",
        "casefold",
        "unicode_nf",
        "map_u_v",
        "map_i_j",
        "strip_diacritics",
        "normalize_ligatures",
    }


def test_config_to_dict_uses_only_canonical_schema():
    config = ensure_app_config(
        {
            "groups": {
                "text": {
                    "files": ["input/*.txt"],
                }
            },
            "nlp": {
                "language": "la",
                "stanza_package": "perseus",
            },
            "filters": {
                "upos_targets": ["NOUN"],
            },
        }
    )

    serialized = config_to_dict(config)

    assert "nlp" in serialized
    assert "filters" in serialized
    assert "exclude_lemmas" not in serialized["filters"]
    assert "prune" not in serialized

    normalization = serialized["normalization"]
    assert {"uv", "ij", "diacritics", "ligatures"}.isdisjoint(normalization)

    for removed in {
        "group",
        "filter",
        "language",
        "stanza_package",
        "stanza_pkg",
        "cpu_only",
        "upos_targets",
        "cleaner_config",
        "vocab_path",
    }:
        assert removed not in serialized


def test_canonical_config_yaml_round_trip():
    original = ensure_app_config(
        {
            "groups": {
                "text": {
                    "files": ["input/*.txt"],
                }
            },
            "nlp": {
                "backend": "stanza",
                "language": "la",
                "stanza_package": "perseus",
                "cpu_only": True,
            },
            "filters": {
                "upos_targets": ["NOUN"],
                "min_token_length": 2,
            },
        }
    )

    serialized = config_to_dict(original)
    serialized_yaml = yaml.safe_dump(serialized, sort_keys=False, allow_unicode=True)
    loaded = yaml.safe_load(serialized_yaml)

    assert ensure_app_config(loaded) == original


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("groups:\n  corpus_a:\n    files: input/*.txt\n", "groups.corpus_a.files"),
        ("groups:\n  corpus_a:\n    files: [input/a.txt, 1]\n", "groups.corpus_a.files.1"),
        ("groups:\n  corpus_a: {files: [input/a.txt]}\nanalysis_unit: token\n", "analysis_unit"),
        (
            "groups:\n  corpus_a: {files: [input/a.txt]}\nfilters:\n  min_token_length: -1\n",
            "filters.min_token_length",
        ),
        (
            "groups:\n  corpus_a: {files: [input/a.txt]}\nfilters:\n  min_token_length: true\n",
            "filters.min_token_length",
        ),
        (
            "groups:\n  corpus_a: {files: [input/a.txt]}\nfilters:\n  upos_targets: NOUN\n",
            "filters.upos_targets",
        ),
        ("groups:\n  corpus_a: {files: [input/a.txt]}\ntrace:\n  max_rows: -1\n", "trace.max_rows"),
    ],
)
def test_load_config_rejects_invalid_typed_values(tmp_path: Path, body: str, match: str):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_config(cfg_path)


def test_load_config_rejects_top_level_non_mapping(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text("- bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Top-level YAML"):
        load_config(cfg_path)


def test_repository_config_is_canonical():
    cfg = load_config(Path("config/groups.config.yml"))

    assert isinstance(cfg, AppConfig)
    assert cfg.grouping.mode == "groups"
    assert cfg.grouping.auto_group_name == "text"
    assert cfg.nlp.backend == "transformers"
    assert cfg.nlp.model_name == "pranaydeep/latin-bert"
    assert cfg.nlp.language == "la"
    assert cfg.nlp.stanza_package == "perseus"
    assert cfg.nlp.cpu_only is True
    assert cfg.analysis_unit == "lemma"
    assert cfg.filters.upos_targets == frozenset({"NOUN", "PROPN"})
    assert cfg.filters.min_token_length == 2
    assert cfg.filters.drop_roman_numerals is True
    assert cfg.normalization.enabled is True
    assert cfg.trace.enabled is True
    assert cfg.trace.path == "output/trace.tsv"
    assert cfg.trace.only_keys == frozenset()
    assert cfg.artifacts.tokens.enabled is False
    assert cfg.archive.enabled is False
    assert set(cfg.groups) == {
        "satyricon_full",
        "satyricon_cena",
        "satyricon_non_cena",
    }


def test_repository_config_uses_only_canonical_top_level_keys():
    raw = yaml.safe_load(
        Path("config/groups.config.yml").read_text(encoding="utf-8")
    )
    assert isinstance(raw, dict)
    schema = AppConfig.model_json_schema(by_alias=True)
    assert set(raw) <= set(schema["properties"])


def test_repository_config_round_trips_through_python_and_yaml():
    original = load_config(Path("config/groups.config.yml"))
    serialized = config_to_dict(original)
    assert ensure_app_config(serialized) == original

    dumped = yaml.safe_dump(serialized, sort_keys=False, allow_unicode=True)
    assert ensure_app_config(yaml.safe_load(dumped)) == original


def test_repository_config_references_existing_files():
    project_root = Path(".").resolve()
    cfg = load_config(Path("config/groups.config.yml"))
    references = (
        cfg.preprocess.config,
        cfg.ref_tags.patterns,
        cfg.dictcheck.wordlist,
        cfg.dictcheck.lemma_normalize,
        cfg.filters.roman_exceptions_file,
    )

    for value in references:
        assert value is not None
        assert (project_root / value).is_file(), value
