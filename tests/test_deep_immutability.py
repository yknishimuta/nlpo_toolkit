from collections import Counter

import pytest

from nlpo_toolkit.comparison.models import FrequencyTable
from nlpo_toolkit.comparison.results import MultiComparisonRow
from nlpo_toolkit.corpus_analysis.analysis_cache.stats import AnalysisCacheStatsCollector
from nlpo_toolkit.corpus_analysis.analysis_results import AnalysisResults, GroupAnalysisResult
from nlpo_toolkit.corpus_analysis.config.models import AppConfig
from nlpo_toolkit.corpus_analysis.partition_validation import PartitionMismatch, PartitionResult
from nlpo_toolkit.corpus_analysis.token_artifact.schema import TokenArtifactNLPDescriptor
from nlpo_toolkit.immutable_collections import freeze_count_mapping, freeze_tuple_mapping


def test_immutable_collection_helpers_copy_sources_and_nested_iterables() -> None:
    counts = {"arma": 2}
    frozen_counts = freeze_count_mapping(counts)
    counts["arma"] = 9
    assert frozen_counts["arma"] == 2
    assert not hasattr(frozen_counts, "update")
    with pytest.raises(TypeError):
        frozen_counts["arma"] = 3  # type: ignore[index]

    files = ["a.txt"]
    frozen_files = freeze_tuple_mapping({"g": files})
    files.append("b.txt")
    assert frozen_files["g"] == ("a.txt",)


def test_config_owned_mappings_are_frozen_but_dump_as_plain_values() -> None:
    groups = {"g": {"files": ["input/*.txt"]}}
    packages = {"tokenize": "default", "pos": "perseus"}
    config = AppConfig.model_validate(
        {"groups": groups, "nlp": {"stanza_package": packages}}
    )
    groups["other"] = {"files": ["other.txt"]}
    packages["pos"] = "changed"

    assert set(config.groups) == {"g"}
    assert config.nlp.stanza_package["pos"] == "perseus"  # type: ignore[index]
    with pytest.raises(TypeError):
        config.groups["other"] = config.groups["g"]  # type: ignore[index]
    dumped = config.model_dump(mode="json")
    assert isinstance(dumped["groups"], dict)
    assert dumped["nlp"]["stanza_package"] == {"tokenize": "default", "pos": "perseus"}


def test_analysis_results_copy_counters_groups_and_cache_snapshot() -> None:
    source = Counter({"arma": 2})
    group = GroupAnalysisResult(files=(), counter=source, ref_tag_counts=source)
    collector = AnalysisCacheStatsCollector(enabled=True, directory="cache")
    results = AnalysisResults.from_groups((("g", group),), cache_stats=collector.snapshot())
    source["arma"] = 10
    collector.record_group(group="g", status="miss", cache_key="key", record_count=1)

    assert results.groups["g"].counter["arma"] == 2
    assert results.cache_stats.misses == 0
    assert not hasattr(results.groups["g"].counter, "subtract")
    with pytest.raises(TypeError):
        results.groups["g"].counter["arma"] = 3  # type: ignore[index]


def test_partition_and_comparison_models_defensively_freeze_inputs() -> None:
    part_counts = {"part": 1}
    mismatch = PartitionMismatch("arma", 2, part_counts, 1, 1, "missing_from_parts")
    rows = [mismatch]
    result = PartitionResult("p", "whole", ["part"], False, 2, 1, 1, 1, 1, 1, 0, rows)
    part_counts["part"] = 9
    rows.clear()
    assert result.parts == ("part",)
    assert result.mismatches == (mismatch,)
    assert mismatch.part_counts["part"] == 1

    counts = {"arma": 2.0}
    table = FrequencyTable("g", counts, 2.0)
    rates = {"g": 1.0}
    row = MultiComparisonRow("arma", counts, rates, 2.0, "g", 1.0, "g", 1.0, 0.0)
    counts["arma"] = 99.0
    rates["g"] = 99.0
    assert table.counts["arma"] == 2.0
    assert row.counts["arma"] == 2.0
    assert row.rates["g"] == 1.0


def test_token_artifact_package_mapping_is_read_only_and_serializable() -> None:
    source = {"tokenize": "default"}
    descriptor = TokenArtifactNLPDescriptor(package=source)
    source["tokenize"] = "changed"
    assert descriptor.package["tokenize"] == "default"  # type: ignore[index]
    with pytest.raises(TypeError):
        descriptor.package["tokenize"] = "changed"  # type: ignore[index]
    assert descriptor.model_dump(mode="json")["package"] == {"tokenize": "default"}
