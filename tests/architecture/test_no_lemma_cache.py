from __future__ import annotations

from pathlib import Path

from nlpo_toolkit.corpus_analysis.config import AppConfig


def test_removed_cache_module_does_not_exist() -> None:
    removed_module = Path("nlpo_toolkit") / "corpus_analysis" / ("lemma" + "_cache.py")

    assert not removed_module.exists()


def test_production_code_has_no_removed_cache_references() -> None:
    removed_name = "lemma" + "_cache"
    offenders: list[Path] = []

    for path in Path("nlpo_toolkit").rglob("*.py"):
        if removed_name in path.read_text(encoding="utf-8"):
            offenders.append(path)

    assert offenders == []


def test_app_config_has_only_analysis_cache() -> None:
    field_names = set(AppConfig.model_fields)

    assert "analysis_cache" in field_names
    assert ("lemma" + "_cache") not in field_names


def test_analysis_cache_imports_without_removed_cache_module() -> None:
    from nlpo_toolkit.corpus_analysis import analysis_cache

    assert analysis_cache is not None
