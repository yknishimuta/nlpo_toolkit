from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import MappingProxyType

import pytest

from nlpo_toolkit.corpus_analysis.config import ensure_app_config
from nlpo_toolkit.corpus_analysis.config_references import ResolvedConfigFiles
from nlpo_toolkit.corpus_analysis.corpus import CorpusWorkItem
from nlpo_toolkit.corpus_analysis.planning.models import (
    AnalysisMode,
    AnalysisPlan,
    ResolvedAnalysisPlan,
)


def _definition(tmp_path: Path, unit="lemma") -> AnalysisPlan:
    config = ensure_app_config({"groups": {"g": {"files": ["input.txt"]}}, "analysis_unit": unit})
    return AnalysisPlan(
        tmp_path,
        tmp_path / "config.yml",
        config,
        tmp_path / "output",
        "groups",
        False,
        AnalysisMode(unit, ("lemma", "count") if unit == "lemma" else ("word", "frequency")),
        None,
        ResolvedConfigFiles(),
    )


def test_analysis_mode_derives_lemma_behavior() -> None:
    assert AnalysisMode("lemma", ("lemma", "count")).use_lemma is True
    assert AnalysisMode("surface", ("word", "frequency")).use_lemma is False


def test_models_are_frozen_and_paths_are_absolute(tmp_path: Path) -> None:
    definition = _definition(tmp_path)
    with pytest.raises(FrozenInstanceError):
        definition.out_dir = tmp_path  # type: ignore[misc]
    assert definition.project_root.is_absolute()
    assert definition.config_path.is_absolute()
    assert definition.out_dir.is_absolute()


def test_resolved_collections_are_immutable_and_no_proxy_fields_exist(tmp_path: Path) -> None:
    item = CorpusWorkItem("g", (tmp_path / "input.txt",))
    resolved = ResolvedAnalysisPlan(_definition(tmp_path), None, [item], {"g": [tmp_path / "input.txt"]})  # type: ignore[arg-type]
    assert resolved.work_items == (item,)
    assert isinstance(resolved.group_files, MappingProxyType)
    assert resolved.group_files["g"] == (tmp_path / "input.txt",)
    with pytest.raises(TypeError):
        resolved.group_files["x"] = ()  # type: ignore[index]
    assert {field.name for field in fields(ResolvedAnalysisPlan)} == {
        "definition", "cleaned_dir", "work_items", "group_files"
    }
    assert "cleaner_inspection" not in {field.name for field in fields(AnalysisPlan)}
