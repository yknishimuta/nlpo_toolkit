from pathlib import Path

from nlpo_toolkit.corpus_analysis.config import (
    AppConfig,
    FilterConfig,
    NormalizationConfig,
    AnalysisCacheConfig,
)
from nlpo_toolkit.comparison.configured import ComparisonSpec
from nlpo_toolkit.corpus_analysis.partition_models import PartitionSpec


def test_unused_config_fields_are_removed() -> None:
    app_fields = set(AppConfig.model_fields)
    filter_fields = set(FilterConfig.model_fields)
    normalization_fields = set(NormalizationConfig.model_fields)

    assert "vocab_path" not in app_fields
    assert "prune" not in app_fields
    assert "exclude_lemmas" not in filter_fields
    assert {"uv", "ij", "diacritics", "ligatures"}.isdisjoint(normalization_fields)
    assert {"use_manifest", "manifest_key_mode"}.isdisjoint(
        AnalysisCacheConfig.model_fields
    )
    assert "report" not in ComparisonSpec.model_fields
    assert "report" in PartitionSpec.model_fields


def test_removed_config_fields_have_no_production_references() -> None:
    removed_fragments = {
        "config.vocab_path",
        "config.filters.exclude_lemmas",
        "config.normalization.uv",
        "config.normalization.ij",
        "config.normalization.diacritics",
        "config.normalization.ligatures",
        "config.prune",
        "config.analysis_cache.use_manifest",
        "config.analysis_cache.manifest_key_mode",
    }
    offenders = []

    for path in Path("nlpo_toolkit").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for fragment in removed_fragments:
            if fragment in text:
                offenders.append((str(path), fragment))

    assert offenders == []
