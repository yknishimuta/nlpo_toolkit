from pathlib import Path

import yaml


REMOVED_TOP_LEVEL_KEYS = {
    "cleaner_config",
    "cpu_only",
    "filter",
    "group",
    "language",
    "lemma_cache",
    "prune",
    "stanza_package",
    "stanza_pkg",
    "upos_targets",
    "vocab_path",
}


def _repository_config() -> dict[str, object]:
    raw = yaml.safe_load(
        Path("config/groups.config.yml").read_text(encoding="utf-8")
    )
    assert isinstance(raw, dict)
    return raw


def test_repository_config_has_no_removed_top_level_keys() -> None:
    assert REMOVED_TOP_LEVEL_KEYS.isdisjoint(_repository_config())


def test_repository_config_has_no_removed_nested_keys() -> None:
    raw = _repository_config()
    filters = raw.get("filters", {})
    ref_tags = raw.get("ref_tags", {})
    normalization = raw.get("normalization", {})
    analysis_cache = raw.get("analysis_cache", {})
    comparisons = raw.get("comparisons", [])
    assert isinstance(filters, dict)
    assert isinstance(ref_tags, dict)
    assert isinstance(normalization, dict)
    assert isinstance(analysis_cache, dict)
    assert isinstance(comparisons, list)

    assert {"roman_exception_files", "exclude_lemmas", "exclude_lemmas_file"}.isdisjoint(filters)
    assert "ref_tags_file" not in ref_tags
    assert {"uv", "ij", "diacritics", "ligatures"}.isdisjoint(normalization)
    assert {"use_manifest", "manifest_key_mode"}.isdisjoint(analysis_cache)
    assert all(isinstance(item, dict) and "report" not in item for item in comparisons)


def test_repository_config_contains_no_commented_alternative_blocks() -> None:
    source = Path("config/groups.config.yml").read_text(encoding="utf-8")
    assert "#groups:" not in source
    assert "#grouping:" not in source
    assert "#path:" not in source
    assert "#only_keys:" not in source
