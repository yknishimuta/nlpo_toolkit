from __future__ import annotations

import math

import pytest
from pydantic import BaseModel, ValidationError

from nlpo_toolkit.comparison.configured import (
    ComparisonSortConfig,
    ComparisonSpec,
)
from nlpo_toolkit.corpus_analysis.config import (
    AnalysisCacheConfig,
    AppConfig,
    ArchiveConfig,
    ArtifactsConfig,
    DictCheckConfig,
    FilterConfig,
    GroupConfig,
    GroupingConfig,
    NLPConfig,
    NormalizationConfig,
    PreprocessConfig,
    RefTagsConfig,
    TokenArtifactConfig,
    TraceConfig,
    ValidationsConfig,
    parse_config,
)
from nlpo_toolkit.corpus_analysis.partition_models import PartitionSpec


CONFIG_MODELS = (
    GroupConfig,
    PreprocessConfig,
    GroupingConfig,
    NLPConfig,
    FilterConfig,
    NormalizationConfig,
    DictCheckConfig,
    RefTagsConfig,
    TraceConfig,
    TokenArtifactConfig,
    ArtifactsConfig,
    ArchiveConfig,
    AnalysisCacheConfig,
    ValidationsConfig,
    AppConfig,
    ComparisonSortConfig,
    ComparisonSpec,
    PartitionSpec,
)


def minimal_raw() -> dict[str, object]:
    return {"groups": {"a": {"files": ["input/a.txt"]}}}


def groups_raw() -> dict[str, object]:
    return {
        "groups": {
            "whole": {"files": ["input/whole.txt"]},
            "a": {"files": ["input/a.txt"]},
            "b": {"files": ["input/b.txt"]},
        }
    }


def test_all_configuration_models_are_frozen_pydantic_models() -> None:
    for model in CONFIG_MODELS:
        assert issubclass(model, BaseModel)
        assert model.model_config["frozen"] is True
        assert model.model_config["extra"] == "forbid"


def test_configuration_instances_are_immutable() -> None:
    config = parse_config(minimal_raw())
    with pytest.raises(ValidationError):
        config.out_dir = "changed"  # type: ignore[misc]


def test_unknown_keys_are_rejected_at_top_and_nested_levels() -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_config({**minimal_raw(), "unknown": True})
    with pytest.raises(ValueError, match="groups.a.unknown"):
        parse_config({"groups": {"a": {"files": ["a.txt"], "unknown": True}}})


def test_removed_cache_and_comparison_fields_are_rejected() -> None:
    assert "use_manifest" not in AnalysisCacheConfig.model_fields
    assert "manifest_key_mode" not in AnalysisCacheConfig.model_fields
    assert "report" not in ComparisonSpec.model_fields
    assert "report" in PartitionSpec.model_fields
    for field, value in (("use_manifest", False), ("manifest_key_mode", "relative")):
        with pytest.raises(ValidationError):
            AnalysisCacheConfig.model_validate({field: value})
    with pytest.raises(ValidationError):
        ComparisonSpec.model_validate({
            "name": "a_vs_b", "group_a": "a", "group_b": "b", "report": "all"
        })


def test_yaml_collections_are_normalized_without_scalar_coercion() -> None:
    raw = minimal_raw()
    raw["filters"] = {"upos_targets": ["NOUN", "propn"]}
    raw["trace"] = {"only_keys": ["lemma", "surface"]}
    config = parse_config(raw)
    assert config.groups["a"].files == ("input/a.txt",)
    assert config.filters.upos_targets == frozenset({"NOUN", "PROPN"})
    assert config.trace.only_keys == frozenset({"LEMMA", "SURFACE"})

    for invalid in ("true", 1):
        with pytest.raises(ValueError):
            parse_config({**minimal_raw(), "dictcheck": {"enabled": invalid}})
    for invalid in (True, 1.0):
        with pytest.raises(ValueError):
            parse_config({**minimal_raw(), "filters": {"min_token_length": invalid}})


def test_blank_required_strings_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        parse_config({"groups": {"   ": {"files": ["a.txt"]}}})
    with pytest.raises(ValueError, match="non-empty string"):
        GroupConfig(files=("   ",))


def test_analysis_cache_alias_has_separate_external_and_python_names() -> None:
    raw = minimal_raw()
    raw["analysis_cache"] = {"dir": ".cache"}
    config = parse_config(raw)
    assert config.analysis_cache.directory == ".cache"
    assert AnalysisCacheConfig(directory="python-cache").directory == "python-cache"
    assert config.analysis_cache.model_dump(by_alias=True)["dir"] == ".cache"
    assert "directory" not in config.analysis_cache.model_dump(by_alias=True)

    with pytest.raises(ValueError, match="analysis_cache.directory"):
        parse_config({**minimal_raw(), "analysis_cache": {"directory": ".cache"}})


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ({}, True),
        ({"kind": "cleaner", "config": "config/cleaner.yml"}, True),
        ({"kind": "cleaner"}, False),
        ({"config": "config/cleaner.yml"}, False),
        ({"kind": "other", "config": "x.yml"}, False),
    ],
)
def test_preprocess_states(value: dict[str, object], valid: bool) -> None:
    if valid:
        PreprocessConfig.model_validate(value)
    else:
        with pytest.raises(ValidationError):
            PreprocessConfig.model_validate(value)


def test_nlp_validation_is_strict_and_preserves_null_package_behavior() -> None:
    assert NLPConfig(stanza_package=None).stanza_package == "perseus"
    assert NLPConfig(
        backend="transformers", model_name="example/model"
    ).model_name == "example/model"
    with pytest.raises(ValidationError, match="model_name is required"):
        NLPConfig(backend="transformers")
    with pytest.raises(ValidationError):
        NLPConfig(stanza_package={1: "perseus"})  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        NLPConfig(cpu_only=1)  # type: ignore[arg-type]


def test_comparison_constraints_and_nested_sort() -> None:
    spec = ComparisonSpec(name=" x ", group_a=" a ", group_b=" b ")
    assert (spec.name, spec.group_a, spec.group_b) == ("x", "a", "b")
    assert spec.sort.by == "log_likelihood"

    invalid_updates = (
        {"group_b": "a"},
        {"scale": 0},
        {"scale": 10_000.0},
        {"zero_correction": 0},
        {"zero_correction": math.inf},
        {"zero_correction": math.nan},
        {"min_total_count": 0},
        {"sort": {"by": "unknown"}},
        {"sort": {"unknown": True}},
    )
    base = {"name": "x", "group_a": "a", "group_b": "b"}
    for update in invalid_updates:
        with pytest.raises(ValidationError):
            ComparisonSpec.model_validate({**base, **update})


def test_partition_constraints() -> None:
    base = {"name": "p", "whole": "whole", "parts": ["a", "b"]}
    assert PartitionSpec.model_validate(base).parts == ("a", "b")
    for update in (
        {"parts": ["a"]},
        {"parts": ["a", "a"]},
        {"parts": ["whole", "a"]},
    ):
        with pytest.raises(ValidationError):
            PartitionSpec.model_validate({**base, **update})


def test_app_config_cross_section_validation() -> None:
    raw = groups_raw()
    raw["comparisons"] = [
        {"name": "c", "group_a": "a", "group_b": "b"},
        {"name": "c", "group_a": "a", "group_b": "whole"},
    ]
    with pytest.raises(ValueError, match="duplicate comparison name"):
        parse_config(raw)

    for section in (
        {"comparisons": [{"name": "c", "group_a": "missing", "group_b": "b"}]},
        {
            "validations": {
                "partitions": [
                    {"name": "p", "whole": "whole", "parts": ["a", "missing"]}
                ]
            }
        },
    ):
        assert parse_config({**groups_raw(), **section})


def test_yaml_per_file_rejects_count_only_specs_but_groups_accepts_them() -> None:
    comparison = {"comparisons": [{"name": "c", "group_a": "a", "group_b": "b"}]}
    partition = {
        "validations": {
            "partitions": [
                {"name": "p", "whole": "whole", "parts": ["a", "b"]}
            ]
        }
    }
    for section in (comparison, partition):
        parse_config({**groups_raw(), **section})
        with pytest.raises(ValueError, match="per_file"):
            parse_config(
                {**groups_raw(), "grouping": {"mode": "per_file"}, **section}
            )
