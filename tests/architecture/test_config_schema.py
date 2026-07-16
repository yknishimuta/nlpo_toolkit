from __future__ import annotations

from pathlib import Path
import json


def test_config_parser_has_no_removed_top_level_fallbacks() -> None:
    source = Path("nlpo_toolkit/corpus_analysis/config/parser.py").read_text(encoding="utf-8")

    forbidden = [
        'raw.get("filter")',
        'raw.get("group")',
        'raw.get("language")',
        'raw.get("stanza_package")',
        'raw.get("stanza_pkg")',
        'raw.get("cpu_only")',
        'raw.get("upos_targets")',
    ]

    for fragment in forbidden:
        assert fragment not in source


def test_production_code_has_no_removed_config_field_references() -> None:
    forbidden = [
        "config." + "vocab_path",
        "config.normalization." + "uv",
        "config.normalization." + "ij",
        "config.normalization." + "diacritics",
        "config.normalization." + "ligatures",
        "config.filters." + "exclude_lemmas",
    ]
    offenders: list[tuple[Path, str]] = []

    for path in Path("nlpo_toolkit").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            if fragment in text:
                offenders.append((path, fragment))

    assert offenders == []


def test_generated_schema_excludes_removed_fields_but_keeps_partition_report() -> None:
    schema = json.loads(
        Path("config/groups.config.schema.json").read_text(encoding="utf-8")
    )
    definitions = schema["$defs"]
    assert {"use_manifest", "manifest_key_mode"}.isdisjoint(
        definitions["AnalysisCacheConfig"]["properties"]
    )
    assert "report" not in definitions["ComparisonSpec"]["properties"]
    assert "report" in definitions["PartitionSpec"]["properties"]
    assert "ComparisonReport" not in definitions
