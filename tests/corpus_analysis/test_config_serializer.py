from __future__ import annotations

from nlpo_toolkit.corpus_analysis.config import config_to_dict, parse_config


def test_external_serialization_is_nested_aliased_deterministic_and_round_trips() -> None:
    raw = {
        "groups": {
            "whole": {"files": ["input/whole.txt"]},
            "a": {"files": ["input/a.txt"]},
            "b": {"files": ["input/b.txt"]},
        },
        "preprocess": {"kind": "cleaner", "config": "config/cleaner.yml"},
        "nlp": {"stanza_package": {"tokenize": "perseus", "pos": "perseus"}},
        "filters": {"upos_targets": ["PROPN", "NOUN"]},
        "normalization": {"casefold": True},
        "dictcheck": {"enabled": True, "wordlist": "data/words.txt"},
        "ref_tags": {"enabled": True, "patterns": "config/ref-tags.txt"},
        "trace": {"only_keys": ["surface", "lemma"]},
        "artifacts": {"tokens": {"enabled": True, "path": "output/tokens.tsv"}},
        "archive": {"enabled": True, "include_input": True},
        "analysis_cache": {"dir": ".cache"},
        "analysis_unit": "surface",
        "csv_header": ["Item", "Count"],
        "comparisons": [
            {
                "name": "a-b",
                "group_a": "a",
                "group_b": "b",
                "sort": {"by": "item", "descending": False},
            }
        ],
        "validations": {
            "partitions": [
                {"name": "whole-parts", "whole": "whole", "parts": ["a", "b"]}
            ]
        },
    }
    config = parse_config(raw)
    first = config_to_dict(config)
    second = config_to_dict(config)

    assert first == second
    assert first["analysis_cache"]["dir"] == ".cache"  # type: ignore[index]
    assert "directory" not in first["analysis_cache"]  # type: ignore[operator]
    assert first["filters"]["upos_targets"] == ["NOUN", "PROPN"]  # type: ignore[index]
    assert first["trace"]["only_keys"] == ["LEMMA", "SURFACE"]  # type: ignore[index]
    assert first["comparisons"][0]["sort"] == {  # type: ignore[index]
        "by": "item",
        "descending": False,
    }
    assert first["validations"]["partitions"][0]["parts"] == ["a", "b"]  # type: ignore[index]
    assert parse_config(first) == config


def test_optional_external_sections_are_omitted_consistently() -> None:
    dumped = config_to_dict(parse_config({"groups": {"a": {"files": ["a.txt"]}}}))
    assert "preprocess" not in dumped
    assert "csv_header" not in dumped
    assert dumped["filters"]["roman_exceptions_file"] is None  # type: ignore[index]
